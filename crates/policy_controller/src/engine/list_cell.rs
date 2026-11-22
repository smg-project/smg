//! The cell a program keeps beside each worker list it publishes, and the
//! engine's value of the list in it.
//!
//! A program that decides in its own process keeps its workers in the lists
//! its registry publishes: one shared list of worker objects for a model at
//! each membership change, and the lists the program derives from that one.
//! The engine keeps one value of each list ([`PoolMembers`]). This is where
//! that value lives and how the engine comes to it:
//!
//! - [`ListCell`] is the cell: one beside each list, in the program's own
//!   value of a membership generation, empty when the list is published.
//! - [`LentList`] is a list lent with its cell and with what the program
//!   knows of the list's workers beside their objects: how a program names
//!   a list to the engine.
//! - [`ModelLists`] is the program's side: for a model, each list published
//!   so far, lent so.
//! - [`PolicyEngine::members_of`] is the engine's read of a lent list: the
//!   value in the cell, made first where the cell is empty.
//!
//! The engine keeps no list and no value of one. A value made here lives in
//! its cell, and falls with the list that the cell stands beside.

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use smg_worker::identity::basic::Worker;

use super::{
    policy_engine::{PolicyEngine, PoolMembers},
    view::FixedFacts,
};

/// The cell beside one worker list: empty when the list is published, and
/// from the list's first ask on the engine's value of it ([`PoolMembers`]).
///
/// A program keeps one beside each list it publishes, in its value of a
/// membership generation. It makes the cell empty where it builds that
/// value, which takes no lock and reads no worker. The engine's value in
/// the cell falls with the program's value, as the list does: a membership
/// change publishes the next list with an empty cell of its own. A cell is
/// lent to one engine, with the list it stands beside ([`LentList`]).
///
/// The cell is two words in the program's value, filled or empty. The
/// engine's value stands in an allocation of its own: it is aligned to
/// cache lines of its own, for the word a pick writes, and a cell that held
/// it in place would give the program's value that alignment and that size
/// for every list, asked or not.
#[derive(Debug, Default)]
pub struct ListCell(OnceLock<Box<PoolMembers>>);

/// A worker list lent with its cell and its facts: a shared list the
/// program's registry published, or one the program derived from it, the
/// cell the program keeps beside that list, and what the program knows of
/// the list's workers beside their objects.
///
/// The cell is the one beside `workers` and is lent with no other list: the
/// value in it is of the list that filled it.
#[derive(Clone, Copy, Debug)]
pub struct LentList<'a> {
    /// The list, as the program published it.
    pub workers: &'a Arc<[Arc<dyn Worker>]>,
    /// The cell the program keeps beside the list.
    pub cell: &'a ListCell,
    /// What the program knows of each worker beside its object
    /// ([`FixedFacts`]): one a worker, as many as the list and in its
    /// order. `None` from a program that knows nothing of its workers:
    /// every worker of the list then states the default.
    ///
    /// A program makes them once a list, where it has the list's workers
    /// at hand, and keeps them beside the list for as long as it stands.
    /// They are read where the list's cell is filled, and at no decision.
    pub facts: Option<&'a [FixedFacts]>,
}

/// A program's worker lists by model, as the engine is handed them: each
/// list with its cell and with what the program knows of its workers.
///
/// A program implements it over its registry. The lists, their cells and
/// their facts live in the program's own value of a membership generation,
/// and an implementation lends them from there: nothing is copied for the
/// engine.
pub trait ModelLists: fmt::Debug + Send + Sync + 'static {
    /// Hands `hand` each list of `model`'s workers that the program has
    /// published so far, with the list's cell and facts: the model's shared
    /// list, and each list the program derived from it. A model the program
    /// holds no worker of is handed nothing.
    ///
    /// A program that builds a derived list at its first use hands the lists
    /// built so far. A list built later is not lost: its cell is filled at
    /// the list's first ask ([`PolicyEngine::members_of`]).
    ///
    /// Each list is lent for the length of one call of `hand`. What fills a
    /// model's cells ahead of the model's requests asks each handed list's
    /// value there, off the path of a request. No decision calls this.
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>));
}

impl PolicyEngine {
    /// The engine's value of a lent list: read from the list's cell, and
    /// made into the cell first where it is empty.
    ///
    /// A filled cell costs one read. An empty one is filled here: the value
    /// is made as [`Self::pool_members_with_fixed_facts`] makes it, of the
    /// facts lent with the list, each worker's at its place, and of the
    /// default where the list is lent with none. The fill reads the lent
    /// list and its facts and asks the program nothing. Where several ask
    /// an empty cell at once, one of them makes the value and the others
    /// wait for it and read the same one.
    ///
    /// The fill writes the list's records, so its cost grows with the list,
    /// once a list. A list whose value is asked ahead of its requests
    /// ([`ModelLists::lists_of`]) leaves each request the one read. A request
    /// that asks first pays the fill.
    ///
    /// Facts of another count than the list's workers are an error of the
    /// program, which a build with debug assertions refuses at the ask.
    /// Another build reads the default for a worker past the last fact.
    pub fn members_of<'a>(&self, lent: LentList<'a>) -> &'a PoolMembers {
        debug_assert!(
            lent.facts
                .is_none_or(|facts| facts.len() == lent.workers.len()),
            "a list is lent with one fact a worker, or with none"
        );
        let members = lent.cell.0.get_or_init(|| {
            // The entry asks once a worker, in the list's order.
            let mut stated = lent.facts.unwrap_or_default().iter();
            let facts_of = |_: &Arc<dyn Worker>| stated.next().copied().unwrap_or_default();
            Box::new(self.pool_members_with_fixed_facts(lent.workers, facts_of))
        });
        debug_assert!(
            Arc::ptr_eq(members.workers(), lent.workers),
            "a cell is lent with the list it stands beside"
        );
        members
    }
}

#[cfg(test)]
mod tests {
    #[cfg(debug_assertions)]
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::{ptr, sync::Barrier, thread};

    use smg_worker::{
        registry::{Generation, WorkerRegistry},
        test_support::{ready_worker, worker_with_model_aliases},
        worker::WorkerType,
    };

    use super::*;
    use crate::document::spec::{AppliedSpec, PolicySpec};

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
    /// facts and its cell.
    #[derive(Debug)]
    struct Lists {
        all: Stated,
        all_cell: ListCell,
        decode: OnceLock<Stated>,
        decode_cell: ListCell,
    }

    impl Generation for Lists {
        fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
            Self {
                all: stated_list(members),
                all_cell: ListCell::default(),
                decode: OnceLock::new(),
                decode_cell: ListCell::default(),
            }
        }

        fn members(&self) -> &Arc<[Arc<dyn Worker>]> {
            &self.all.0
        }
    }

    impl Lists {
        fn all(&self) -> LentList<'_> {
            LentList {
                workers: &self.all.0,
                cell: &self.all_cell,
                facts: Some(&self.all.1),
            }
        }

        /// The decode workers: a list built here, at its first use.
        fn decode(&self) -> LentList<'_> {
            let (workers, facts) = self.decode.get_or_init(|| {
                let all = self.all.0.iter();
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

    fn engine() -> PolicyEngine {
        let document = PolicySpec::preset("round_robin").expect("a built-in preset");
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the preset validates"))
    }

    fn regular(at: usize) -> Arc<dyn Worker> {
        ready_worker(&format!("http://worker-{at}:8000"), MODEL)
    }

    /// Whether each worker takes images, as the list's records under
    /// `engine` say it, by the worker's place.
    fn take_images(engine: &PolicyEngine, members: &PoolMembers) -> Vec<bool> {
        let records = members.records(engine.spec(), engine.clock());
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
        assert!(lists.all_cell.0.get().is_none(), "empty at the publish");
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
        assert!(Arc::ptr_eq(first.workers(), &lists.all.0), "over the list");
        assert!(asked.iter().all(|members| ptr::eq(*members, first)));
        assert!(ptr::eq(engine.members_of(lists.all()), first));
        let filled: &PoolMembers = lists.all_cell.0.get().expect("a filled cell");
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
        let old_list = Arc::downgrade(&old.all.0);
        assert_eq!(engine.members_of(old.all()).workers().len(), 2);

        let registered = program.registry.register(regular(2));
        assert!(registered.is_some(), "a new worker registers");
        let new = program.published();
        assert_eq!(new.all.0.len(), 3);
        assert!(
            new.all_cell.0.get().is_none(),
            "the new list's cell is empty"
        );
        assert!(old.all_cell.0.get().is_some(), "the old value stands");
        let members = engine.members_of(new.all());
        assert!(Arc::ptr_eq(members.workers(), &new.all.0));

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
        let decode =
            worker_with_model_aliases("http://decode-0:8000", MODEL, &[], WorkerType::Decode);
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
        assert!(Arc::ptr_eq(&both[0].0, &lists.all.0));
        assert!(ptr::eq(both[0].1, &lists.all_cell));
        assert!(Arc::ptr_eq(&both[1].0, lists.decode().workers));
        assert!(ptr::eq(both[1].1, &lists.decode_cell));

        program.lists_of(MODEL, &mut |lent| {
            let members = engine.members_of(lent);
            assert!(Arc::ptr_eq(members.workers(), lent.workers));
        });
        let filled = |cell: &ListCell| Arc::clone(cell.0.get().expect("a filled cell").workers());
        assert!(Arc::ptr_eq(&filled(&lists.all_cell), &lists.all.0));
        assert!(Arc::ptr_eq(
            &filled(&lists.decode_cell),
            lists.decode().workers
        ));
    }

    /// The facts lent with a list go into the list's records where its cell
    /// is filled, each worker's at its place: the workers the program names
    /// take images and no other does. One list lent beside two cells keeps
    /// each cell's own facts, and a list lent with none says of no worker
    /// that it takes images.
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
    }

    /// What the engine says of an ask that it refuses in a build with debug
    /// assertions.
    #[cfg(debug_assertions)]
    fn refused(engine: &PolicyEngine, lent: LentList<'_>) -> &'static str {
        let asked = catch_unwind(AssertUnwindSafe(|| {
            let _ = engine.members_of(lent);
        }));
        let said = asked.expect_err("the ask is refused");
        said.downcast_ref::<&str>().copied().unwrap_or_default()
    }

    /// A list lent with facts of another count than its workers is refused
    /// at the ask, in a build with debug assertions, and its cell stays
    /// empty.
    #[cfg(debug_assertions)]
    #[test]
    fn facts_of_another_count_than_the_list_are_refused() {
        let (engine, (list, facts)) = (engine(), stated_list((0..3).map(regular).collect()));
        let cell = ListCell::default();
        let lent = LentList {
            workers: &list,
            cell: &cell,
            facts: Some(&facts[..2]),
        };
        assert!(refused(&engine, lent).contains("one fact a worker"));
        assert!(cell.0.get().is_none(), "the cell stays empty");
    }

    /// A cell lent with another list than the one that filled it is
    /// refused at the ask, in a build with debug assertions.
    #[cfg(debug_assertions)]
    #[test]
    fn a_cell_lent_with_another_list_is_refused() {
        let (engine, (list, _)) = (engine(), stated_list((0..2).map(regular).collect()));
        let (other, _) = stated_list((2..4).map(regular).collect());
        let cell = ListCell::default();
        let lent = LentList {
            workers: &list,
            cell: &cell,
            facts: None,
        };
        assert!(Arc::ptr_eq(engine.members_of(lent).workers(), &list));
        let lent = LentList {
            workers: &other,
            ..lent
        };
        assert!(refused(&engine, lent).contains("the list it stands beside"));
    }
}
