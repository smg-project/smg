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
//! - [`LentList`] is a list lent with its cell: how a program names a list
//!   to the engine.
//! - [`ModelLists`] is the program's side: for a model, each list published
//!   so far with its cell, and what the program knows of a worker beside
//!   the worker's object.
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

/// A worker list lent with its cell: a shared list the program's registry
/// published, or one the program derived from it, and the cell the program
/// keeps beside that list.
///
/// The cell is the one beside `workers` and is lent with no other list: the
/// value in it is of the list that filled it.
#[derive(Clone, Copy, Debug)]
pub struct LentList<'a> {
    /// The list, as the program published it.
    pub workers: &'a Arc<[Arc<dyn Worker>]>,
    /// The cell the program keeps beside the list.
    pub cell: &'a ListCell,
}

/// A program's worker lists by model, as the engine is handed them: each
/// list with its cell, and what the program knows of a worker beside the
/// worker's object.
///
/// A program implements it over its registry. The lists and their cells
/// live in the program's own value of a membership generation, and an
/// implementation lends them from there: nothing is copied for the engine.
pub trait ModelLists: fmt::Debug + Send + Sync + 'static {
    /// Hands `hand` each list of `model`'s workers that the program has
    /// published so far, with the list's cell: the model's shared list, and
    /// each list the program derived from it. A model the program holds no
    /// worker of is handed nothing.
    ///
    /// A program that builds a derived list at its first use hands the lists
    /// built so far. A list built later is not lost: its cell is filled at
    /// the list's first ask ([`PolicyEngine::members_of`]).
    ///
    /// Each list is lent for the length of one call of `hand`. What fills a
    /// model's cells ahead of the model's requests asks each handed list's
    /// value there, off the path of a request. No decision calls this.
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>));

    /// What the program knows of `worker` beside its object
    /// ([`FixedFacts`]): whether the worker takes images. A program that
    /// knows no such fact answers the default.
    ///
    /// Asked once a worker where the cell of a list is filled
    /// ([`PolicyEngine::members_of`]), in the list's order, and at no
    /// decision.
    ///
    /// It runs inside the fill of the list's cell, so it asks the engine
    /// nothing of that list: a call of [`PolicyEngine::members_of`] for the
    /// same list from here waits for the fill it stands in, and does not
    /// return.
    fn fixed_facts(&self, worker: &Arc<dyn Worker>) -> FixedFacts;
}

impl PolicyEngine {
    /// The engine's value of a lent list: read from the list's cell, and
    /// made into the cell first where it is empty.
    ///
    /// A filled cell costs one read. An empty one is filled here: the value
    /// is made as [`Self::pool_members_with_fixed_facts`] makes it, with
    /// `lists` asked once a worker for what the program knows of it
    /// ([`ModelLists::fixed_facts`]). Where several ask an empty cell at
    /// once, one of them makes the value and the others wait for it and read
    /// the same one. A fill whose program panics leaves the cell empty: the
    /// panic goes to the asker, and a later ask fills the cell.
    ///
    /// The fill writes the list's records, so its cost grows with the list,
    /// once a list. A list whose value is asked ahead of its requests
    /// ([`ModelLists::lists_of`]) leaves each request the one read. A request
    /// that asks first pays the fill.
    pub fn members_of<'a>(
        &self,
        lent: LentList<'a>,
        lists: &(impl ModelLists + ?Sized),
    ) -> &'a PoolMembers {
        let members = lent.cell.0.get_or_init(|| {
            let facts_of = |worker: &Arc<dyn Worker>| lists.fixed_facts(worker);
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
    use std::{
        ptr,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Barrier,
        },
        thread,
    };

    use smg_worker::{
        registry::{Generation, WorkerRegistry},
        test_support::{ready_worker, worker_with_model_aliases},
        worker::WorkerType,
    };

    use super::*;
    use crate::document::spec::{AppliedSpec, PolicySpec};

    const MODEL: &str = "model-a";

    /// The worker the test program says takes images.
    const TAKES_IMAGES: &str = "http://worker-1:8000";

    /// A program's value of one generation: the model's shared list, its
    /// decode workers as a list built at its first use, and a cell beside
    /// each.
    #[derive(Debug)]
    struct Lists {
        all: Arc<[Arc<dyn Worker>]>,
        all_cell: ListCell,
        decode: OnceLock<Arc<[Arc<dyn Worker>]>>,
        decode_cell: ListCell,
    }

    impl Generation for Lists {
        fn of(members: Arc<[Arc<dyn Worker>]>) -> Self {
            Self {
                all: members,
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
            LentList {
                workers: &self.all,
                cell: &self.all_cell,
            }
        }

        /// The decode workers: a list built here, at its first use.
        fn decode(&self) -> LentList<'_> {
            let workers = self.decode.get_or_init(|| {
                let all = self.all.iter();
                all.filter(|worker| *worker.worker_type() == WorkerType::Decode)
                    .cloned()
                    .collect()
            });
            LentList {
                workers,
                cell: &self.decode_cell,
            }
        }
    }

    /// A program whose lists live in the worker crate's registry. It says
    /// of one worker that it takes images, and counts what it is asked.
    #[derive(Debug)]
    struct Program {
        registry: WorkerRegistry<Lists>,
        asked: AtomicUsize,
    }

    impl Program {
        fn of(workers: &[Arc<dyn Worker>]) -> Self {
            let registry = WorkerRegistry::new();
            for worker in workers {
                let registered = registry.register(Arc::clone(worker));
                assert!(registered.is_some(), "a new worker registers");
            }
            Self {
                registry,
                asked: AtomicUsize::new(0),
            }
        }

        /// The model's generation as the registry holds it now.
        fn published(&self) -> Arc<Lists> {
            let published = self.registry.model_generation(MODEL);
            published.expect("the model has workers")
        }

        fn asked(&self) -> usize {
            self.asked.load(Ordering::Relaxed)
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

        fn fixed_facts(&self, worker: &Arc<dyn Worker>) -> FixedFacts {
            self.asked.fetch_add(1, Ordering::Relaxed);
            FixedFacts {
                vision: worker.url() == TAKES_IMAGES,
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

    /// A list's cell is empty when the list is published. The first ask
    /// fills it with the engine's value of that list, and every later ask
    /// reads the same value. Where several threads ask an empty cell at
    /// once, one value is made: the program is asked once a worker and no
    /// more. The cell is two words beside the list.
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
                engine.members_of(lists.all(), &program)
            };
            let asks: Vec<_> = (0..4).map(|_| scope.spawn(ask)).collect();
            let joined = asks.into_iter().map(|ask| ask.join());
            joined.map(|ask| ask.expect("an ask ends")).collect()
        });
        let first = asked[0];
        assert!(Arc::ptr_eq(first.workers(), &lists.all), "over the list");
        assert!(asked.iter().all(|members| ptr::eq(*members, first)));
        assert!(ptr::eq(engine.members_of(lists.all(), &program), first));
        assert_eq!(program.asked(), 3, "one fill, which asks once a worker");
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
        assert_eq!(engine.members_of(old.all(), &program).workers().len(), 2);

        let registered = program.registry.register(regular(2));
        assert!(registered.is_some(), "a new worker registers");
        let new = program.published();
        assert_eq!(new.all.len(), 3);
        assert!(
            new.all_cell.0.get().is_none(),
            "the new list's cell is empty"
        );
        assert!(old.all_cell.0.get().is_some(), "the old value stands");
        let members = engine.members_of(new.all(), &program);
        assert!(Arc::ptr_eq(members.workers(), &new.all));

        drop(old);
        assert!(old_list.upgrade().is_none(), "the old list fell whole");
    }

    /// A program hands each list of a model that it has published so far,
    /// each with its own cell: the model's shared list at once, and a list
    /// it builds at its first use from then on. A model it holds no worker
    /// of is handed nothing. The value asked of each handed list fills that
    /// list's own cell.
    #[test]
    fn each_list_of_a_model_is_handed_with_its_own_cell() {
        let decode =
            worker_with_model_aliases("http://decode-0:8000", MODEL, &[], WorkerType::Decode);
        let (engine, program) = (engine(), Program::of(&[regular(0), regular(1), decode]));
        let lists = program.published();
        let handed = |model: &str| {
            let mut seen = Vec::new();
            program.lists_of(model, &mut |lent| {
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
            let members = engine.members_of(lent, &program);
            assert!(Arc::ptr_eq(members.workers(), lent.workers));
        });
        let filled = |cell: &ListCell| Arc::clone(cell.0.get().expect("a filled cell").workers());
        assert!(Arc::ptr_eq(&filled(&lists.all_cell), &lists.all));
        assert!(Arc::ptr_eq(
            &filled(&lists.decode_cell),
            lists.decode().workers
        ));
    }

    /// What the program knows of a worker is asked where a list's cell is
    /// filled, once a worker, and goes into the list's records: the worker
    /// the program names takes images and no other does. Nothing is asked
    /// where the list is published, and nothing again where the value or its
    /// records are read.
    #[test]
    fn the_programs_facts_are_asked_once_a_worker_at_the_fill() {
        let workers = [regular(0), regular(1), regular(2), regular(3)];
        let (engine, program) = (engine(), Program::of(&workers));
        let lists = program.published();
        assert_eq!(program.asked(), 0, "nothing is asked at the publish");

        let members = engine.members_of(lists.all(), &program);
        assert_eq!(program.asked(), 4, "asked at the fill, once a worker");
        let records = members.records(engine.spec(), engine.clock());
        let takes_images: Vec<bool> = records.fixed().iter().map(|row| row.vision).collect();
        let named: Vec<bool> = lists
            .all
            .iter()
            .map(|worker| worker.url() == TAKES_IMAGES)
            .collect();
        assert_eq!(takes_images, named);
        assert_eq!(named.iter().filter(|named| **named).count(), 1);
        assert!(ptr::eq(engine.members_of(lists.all(), &program), members));
        assert_eq!(program.asked(), 4, "and not again at a read");
    }
}
