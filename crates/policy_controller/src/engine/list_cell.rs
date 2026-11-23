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
//!   so far, lent so. A program hands it once, beside the engine's start
//!   ([`PolicyEngine::set_lists`]), and the engine's own task fills a
//!   model's cells through it, ahead of the model's requests.
//! - [`PolicyEngine::members_of`] is the engine's read of a lent list: the
//!   value the cell holds under the engine's document, made first where it
//!   holds none.
//!
//! The engine keeps no list and no value of one. A value made here lives in
//! its cell, and falls with the list that the cell stands beside.
//!
//! A value is made under one applied document, by the engine that decides
//! under it. A cell holds a value for each document its list is asked
//! under, and an engine reads the one made under its own.

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use smg_worker::identity::basic::Worker;

use super::{
    policy_engine::{Loaded, PolicyEngine, PoolMembers},
    view::FixedFacts,
};

/// The cell beside one worker list: empty when the list is published, and
/// from the list's first ask on the engine's value of it.
///
/// A program keeps one beside each list it publishes, in its value of a
/// membership generation. It makes the cell empty where it builds that
/// value, which takes no lock and reads no worker. What the cell holds
/// falls with the program's value, as the list does: a membership change
/// publishes the next list with an empty cell of its own. A cell is lent
/// with the list it stands beside ([`LentList`]).
///
/// A value is made under one applied document, by the engine that decides
/// under it, and an engine reads the value made under its own document and
/// no other. So a cell holds one value for
/// each applied document that asked its list, of one engine or of several,
/// until the list is published again. A value is neither freed nor moved
/// while its list stands: a decision under way keeps the one it read.
///
/// The engine asks a cell inside the decision or the pick that is lent
/// its list ([`WorkerLists`](super::decision_engine::WorkerLists),
/// [`PolicyEngine::pick`]), where it decides over the list. The first ask
/// under a document fills the cell: it reads each worker's registration
/// and writes the list's records, so its cost grows with the list. Where
/// the program handed its lists ([`PolicyEngine::set_lists`]), the
/// engine's own task asks first, off the path of a request, behind the
/// registry's word of the membership change. Else, and for a request that
/// comes ahead of the task, the request that asks first pays the fill.
/// Every later ask is one read and one compare of the document's number,
/// and one more of both for each value made before the asker's. A program
/// asks the engine for no value of a list.
///
/// The cell is two words in the program's value, filled or empty. Each
/// value stands in an allocation of its own: it is aligned to cache lines
/// of its own, for the word a pick writes, and a cell that held it in place
/// would give the program's value that alignment and that size for every
/// list, asked or not.
#[derive(Debug, Default)]
pub struct ListCell(OnceLock<Box<Made>>);

impl Drop for ListCell {
    /// The cell's values fall one behind the other, and not each inside the
    /// fall of the one made before it: a cell holds as many values as
    /// documents asked its list, and their fall takes no deeper a stack.
    fn drop(&mut self) {
        let mut next = self.0.take();
        while let Some(mut made) = next {
            next = made.next.0.take();
        }
    }
}

/// One value of a list in its cell: the engine's value, the number of the
/// applied document it was made under, and the cell of the value made
/// behind it under another number.
#[derive(Debug)]
struct Made {
    under: u64,
    members: PoolMembers,
    next: ListCell,
}

impl Made {
    /// The value of a lent list, made by `engine` under the document whose
    /// number is `under`.
    fn of(engine: Loaded<'_, '_>, lent: LentList<'_>, under: u64) -> Self {
        assert!(
            lent.facts
                .is_none_or(|facts| facts.len() == lent.workers.len()),
            "a list is lent with one fact a worker, or with none"
        );
        // The entry asks once a worker, in the list's order.
        let mut stated = lent.facts.unwrap_or_default().iter();
        let facts_of = |_: &Arc<dyn Worker>| stated.next().copied().unwrap_or_default();
        Self {
            under,
            members: engine.pool_members_with_fixed_facts(lent.workers, facts_of),
            next: ListCell::default(),
        }
    }
}

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
    /// A program makes them once a list, with the list's workers at hand,
    /// and keeps them beside the list for as long as it stands. Where a
    /// registry makes the program's value of a generation under its own
    /// guard, the program makes them behind that, at the list's first
    /// lending or by what ordered the membership change: no member is read
    /// under the guard. They are read where the list's cell is filled: by
    /// the engine's own task ahead of the list's requests
    /// ([`ModelLists::lists_of`]), or inside the first decision or pick
    /// that asks it, and nowhere else.
    ///
    /// Facts of another count than the list are an error of the program,
    /// refused where the list's value is made, in every build, and the cell
    /// takes no value, so every later ask of the list lent so is refused
    /// again. The refusal is a panic. A pick passes it to its caller
    /// ([`PolicyEngine::pick`]), and so does a pair's or an encode
    /// request's own `decide`, which a host calls itself. A decision
    /// through one of the engine's entries is answered
    /// `DecisionRefused::Faulted` and counted with the engine's faults: the
    /// entry catches the panic. The engine's task holds it and counts it
    /// with its own ([`PolicyEngine::upkeep_faults`]).
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
    /// the list's first ask.
    ///
    /// Each list is lent for the length of one call of `hand`. The
    /// engine's own task calls this, off the path of a request, to fill a
    /// model's cells ahead of the model's requests
    /// ([`PolicyEngine::set_lists`]): `hand` asks each list's value under
    /// the engine's document, and so fills a cell that holds none. No
    /// decision calls this.
    ///
    /// `hand` reads each worker of a list it fills, so an implementation
    /// calls it under no guard of its registry. `hand` does not panic: a
    /// list it refuses is held and counted on the engine's side. The call
    /// runs on the task's own loop, which hears no word and keeps no newer
    /// load snapshot meanwhile, so an implementation does not wait in it.
    ///
    /// What `hand` is handed, it fills with the facts lent. The task asks
    /// behind the registry's word of a membership change, which can be
    /// ahead of the program's own record of that change. So a program that
    /// makes a list's facts of such a record hands the list once the record
    /// stands, and nothing of the model until then: a cell filled with
    /// facts that are not whole keeps them for as long as its list stands.
    fn lists_of(&self, model: &str, hand: &mut dyn FnMut(LentList<'_>));
}

/// The lists a program handed the engine ([`PolicyEngine::set_lists`]).
#[derive(Debug, Default)]
pub(super) struct HandedLists(OnceLock<Arc<dyn ModelLists>>);

impl HandedLists {
    /// The lists, where the program handed any: the engine's task asks
    /// for a word's models only then.
    pub(super) fn get(&self) -> Option<&Arc<dyn ModelLists>> {
        self.0.get()
    }
}

impl PolicyEngine {
    /// Hands the engine its program's worker lists, once, at the program's
    /// start: beside the engine's own start ([`Self::start`]), which does
    /// not take them yet. A later call is ignored.
    ///
    /// The engine's own task then fills a model's cells ahead of the
    /// model's requests ([`Self::run`]), by the registry the program handed
    /// ([`Self::set_registry`]). Behind a word of a registration, a
    /// replacement or a removal it asks the lists of each model the word's
    /// worker serves ([`ModelLists::lists_of`]), and behind a whole read of
    /// the registry the lists of each model a standing worker serves. It
    /// asks each handed list's value under the engine's document. A cell
    /// that holds none is filled there, off the path of a request, and the
    /// request that asks the list first reads a filled cell. Words that
    /// stand ready together are heard first, and their models are asked
    /// behind the last of them, each once. The task's upkeep asks what is
    /// owed at that minute, with words still standing: a model asked there
    /// is asked again behind the last word, and that ask reads the cell.
    ///
    /// The task asks behind a word alone, and it asks the lists that stand
    /// at its ask. So these are filled ahead only where a later ask of
    /// their model finds them, and by their first request otherwise, as
    /// under an engine handed no lists:
    ///
    /// - A list a request asks ahead of the task: the task hears the
    ///   registry's word some time behind the registry's write.
    /// - A list published again ahead of the task's ask: the task fills
    ///   the newer one, and a request that still holds the older fills it.
    /// - A list published with no word of the registry. The worker crate's
    ///   registry publishes each model's list again where a worker's model
    ///   cards are refreshed, and sends no event of it.
    /// - A list of a model its word's worker does not state. A worker
    ///   names its models by its cards of the minute, and the worker
    ///   crate's registry keeps a worker in the list of a model its
    ///   refreshed cards left out, until the worker is removed.
    /// - A list the program lends under no model a worker serves, such as
    ///   a list of every worker for a request that names no model.
    /// - A list the program builds behind the task's ask.
    ///
    /// The task runs the program's `lists_of` and each fill on its own
    /// loop: meanwhile it hears no word and keeps no newer load snapshot.
    /// Each value made counts as a pool made, as one a request makes: a
    /// stage that keeps what it learned of a pool by that count, as the
    /// hash ring's does on its own walk, learns each pool again at its
    /// next keyed request.
    ///
    /// A panic in a fill is held and counted ([`Self::upkeep_faults`]). A
    /// list lent with facts of another count than its workers costs its
    /// own fill alone, and the refusal does not unwind into the program's
    /// code. A panic of the program's own code in `lists_of` costs the
    /// lists of that model it had not handed. The stages are told the word
    /// all the same, and the task goes on.
    ///
    /// Lists handed behind the task's first poll are taken all the same.
    /// At its next upkeep the task asks the lists of every model a standing
    /// worker serves, once, and each word from then on asks its own. So a
    /// caller may rely on this: the lists that stand are filled within one
    /// upkeep of the call, and a request that comes sooner fills its own
    /// list, as under an engine handed no lists. A call ahead of the
    /// task's first poll costs nothing more: the task's first whole read
    /// of the registry asks them.
    ///
    /// An engine handed lists and no registry fills nothing ahead.
    pub fn set_lists(&self, lists: Arc<dyn ModelLists>) {
        // The first hand-over stands.
        drop(self.lists.0.set(lists));
    }
}

impl Loaded<'_, '_> {
    /// The engine's value of a lent list under the document it decides
    /// under: read from the list's cell, and made into the cell first where
    /// the cell holds none under that document.
    ///
    /// A value the cell took first costs one read and one compare of the
    /// document's number, in the caller's own code. One made behind it is
    /// found out of line, at one more of both for each value before it.
    /// Where the cell holds none under the engine's document, the value is
    /// made here, as [`Self::pool_members_with_fixed_facts`] makes it, of
    /// the facts lent with the list, each worker's at its place, and of the
    /// default where the list is lent with none. The fill reads the lent
    /// list and its facts and asks the program nothing. Where several ask
    /// at once for a value the cell does not hold, one of them makes it and
    /// the others wait for it and read the same one.
    ///
    /// A value made under another document stays in the cell beside the new
    /// one, where it stood: a cell asked under a second document, or by a
    /// second engine, holds a value for each.
    ///
    /// The fill reads each worker's registration and writes the list's
    /// records, so its cost grows with the list, once for each document the
    /// list is asked under. The engine's own task pays it ahead of the
    /// list's requests where the program handed its lists
    /// ([`PolicyEngine::set_lists`]). Else the request that asks first
    /// pays it.
    ///
    /// # Panics
    ///
    /// Facts of another count than the list's workers are an error of the
    /// program, refused in every build where the value is made: the ask
    /// panics and the cell takes no value, so each later ask of the list
    /// lent so is refused again, and the list lent right is filled. A
    /// decision's entry catches the panic and answers a fault, and the
    /// engine's task holds it and counts it
    /// ([`PolicyEngine::upkeep_faults`]). The check stands in the fill and
    /// costs a read nothing: a cell that holds the value is read with no
    /// look at the facts.
    #[inline]
    pub(crate) fn members_of<'a>(self, lent: LentList<'a>) -> &'a PoolMembers {
        let number = self.document.applied;
        let members = match lent.cell.0.get() {
            // The value the cell took first is read here, in the caller.
            Some(made) if made.under == number => &made.members,
            // One made behind it is found out of line, and one the cell
            // does not hold is made there.
            _ => self.found_or_made(lent.workers, lent.cell, lent.facts, number),
        };
        debug_assert!(
            Arc::ptr_eq(members.workers(), lent.workers),
            "a cell is lent with the list it stands beside"
        );
        members
    }

    /// [`Self::members_of`] out of line: the value of a lent list under
    /// the document `number`, found along the cell's chain, or made where
    /// the chain holds none. Several may come here at once for one empty
    /// cell: one of them makes the value and the others wait for it. It
    /// takes the lent list's three parts: its caller writes no copy of the
    /// list for a call it does not make.
    #[cold]
    #[inline(never)]
    fn found_or_made<'a>(
        self,
        workers: &'a Arc<[Arc<dyn Worker>]>,
        mut cell: &'a ListCell,
        facts: Option<&'a [FixedFacts]>,
        number: u64,
    ) -> &'a PoolMembers {
        let lent = LentList {
            workers,
            cell,
            facts,
        };
        loop {
            let made = cell
                .0
                .get_or_init(|| Box::new(Made::of(self, lent, number)));
            if made.under == number {
                return &made.members;
            }
            cell = &made.next;
        }
    }
}

impl PolicyEngine {
    /// [`Loaded::members_of`] under the engine's document: for a caller
    /// that asks a list's value outside a decision of the engine, as the
    /// engine's own task does ahead of a list's requests.
    #[inline]
    pub(crate) fn members_of<'a>(&self, lent: LentList<'a>) -> &'a PoolMembers {
        self.loaded().members_of(lent)
    }

    /// The bench's count: how many times the reports of `lent`'s list were
    /// converted under this engine's document so far. It makes no value: a
    /// list the engine made no value of counts zero.
    #[doc(hidden)]
    #[must_use]
    pub fn conversions(&self, lent: LentList<'_>) -> u64 {
        let (number, mut cell) = (self.applied(), lent.cell);
        while let Some(made) = cell.0.get() {
            if made.under == number {
                return made.members.conversions();
            }
            cell = &made.next;
        }
        0
    }
}

#[cfg(test)]
pub(crate) mod tests {
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

    /// A worker list as a case publishes it: the list, the cell beside it,
    /// and what the case states of its workers. What the crate's cases hold
    /// where a program holds its own value of a membership generation.
    #[derive(Debug)]
    pub(crate) struct PublishedList {
        workers: Arc<[Arc<dyn Worker>]>,
        facts: Option<Box<[FixedFacts]>>,
        cell: ListCell,
    }

    impl PublishedList {
        /// A list whose workers state nothing.
        pub(crate) fn new(workers: Arc<[Arc<dyn Worker>]>) -> Self {
            Self {
                workers,
                facts: None,
                cell: ListCell::default(),
            }
        }

        /// A list with what `facts_of` states of each worker: asked once a
        /// worker, in the list's order.
        pub(crate) fn with_facts(
            workers: Arc<[Arc<dyn Worker>]>,
            facts_of: impl FnMut(&Arc<dyn Worker>) -> FixedFacts,
        ) -> Self {
            let facts = workers.iter().map(facts_of).collect();
            Self {
                workers,
                facts: Some(facts),
                cell: ListCell::default(),
            }
        }

        /// The list, lent with its cell and its facts.
        pub(crate) fn lent(&self) -> LentList<'_> {
            LentList {
                workers: &self.workers,
                cell: &self.cell,
                facts: self.facts.as_deref(),
            }
        }

        /// The list.
        pub(crate) fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
            &self.workers
        }

        /// Whether the cell holds a value: whether a decision or the
        /// engine's task asked it.
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

    fn engine() -> PolicyEngine {
        let document = PolicySpec::preset("round_robin").expect("a built-in preset");
        PolicyEngine::new(AppliedSpec::validate(&document).expect("the preset validates"))
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
            let ask = |engine: &'_ PolicyEngine| {
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
        let unasked = ListCell::default();
        let lent = LentList {
            workers: &list,
            cell: &unasked,
            facts: None,
        };
        let mut cell = ListCell::default();
        for under in 0..20_000 {
            let mut made = Box::new(Made::of(engine.loaded(), lent, under));
            made.next = cell;
            cell = ListCell(OnceLock::from(made));
        }
        let small = thread::Builder::new().stack_size(64 * 1024);
        let fall = small.spawn(move || drop(cell)).expect("a thread starts");
        fall.join().expect("the values fall");
    }

    /// What the engine says of an ask that it refuses.
    fn refused(engine: &PolicyEngine, lent: LentList<'_>) -> &'static str {
        let asked = catch_unwind(AssertUnwindSafe(|| {
            let _ = engine.members_of(lent);
        }));
        let said = asked.expect_err("the ask is refused");
        said.downcast_ref::<&str>().copied().unwrap_or_default()
    }

    /// A list lent with fewer facts than its workers, or with more, is
    /// refused where its value is made, in every build, and its cell stays
    /// empty: the same list lent right is filled, of its facts.
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
        for wrong in [&facts[..2], &more[..]] {
            assert!(refused(&engine, lent(wrong)).contains("one fact a worker"));
            assert!(first_of(&cell).is_none(), "the cell stays empty");
        }
        let members = engine.members_of(lent(&facts[..]));
        assert_eq!(take_images(&engine, members), [false, true, true]);
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
