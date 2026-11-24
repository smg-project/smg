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
//! - [`PublishedList`] is a list with its cell and its facts in one value:
//!   what a program keeps of a list it publishes, and lends from.
//! - [`ModelLists`] is the program's side: for a model, each list published
//!   so far, lent so. A program hands it once, beside the engine's start
//!   ([`InProcess::set_lists`]), and the engine's own task fills a
//!   model's cells through it, ahead of the model's requests.
//! - [`InProcess::members_of`] is the engine's read of a lent list: the
//!   value the cell holds under the document in force, made first where
//!   it holds none.
//!
//! The engine keeps no list and no value of one. A value made here lives in
//! its cell, and falls with the list that the cell stands beside.
//!
//! A value is made under one applied document, by the engine that decides
//! under it. A cell holds a value for each document its list is asked
//! under, and an engine reads the one made under its own.

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

use std::{
    fmt,
    sync::{Arc, OnceLock},
};

use smg_worker::identity::basic::Worker;

use super::{
    in_process::{InProcess, Loaded, PoolMembers},
    view::{FixedFacts, WorkerId},
};

/// The cell beside one worker list: empty when the list is published, and
/// from the list's first ask on the value each asking engine made of it.
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
/// Nothing else bounds them. A host that applies a document while its
/// lists stand grows each asked cell by one value an apply, the newest
/// last, and a read walks past each value made before its own. A host for
/// which that matters publishes its lists again behind an apply: the lists
/// published then stand beside empty cells.
///
/// The engine asks a cell inside the decision or the pick that is lent
/// its list ([`WorkerLists`](super::decision_engine::WorkerLists),
/// [`InProcess::pick`]), where it decides over the list. The first ask
/// under a document fills the cell: it reads each worker's registration
/// and writes the list's records, so its cost grows with the list. Where
/// the program handed its lists ([`InProcess::set_lists`]), the
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
#[derive(Default)]
pub struct ListCell(OnceLock<Box<Made>>);

impl fmt::Debug for ListCell {
    /// How many values the cell holds and the number of the document each
    /// was made under, in the order they were made. No worker's record is
    /// printed: a program's value of a generation is formatted with its
    /// cells, and a cell's values hold every record of its list.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut under = Vec::new();
        let mut cell = self;
        while let Some(made) = cell.0.get() {
            under.push(made.under);
            cell = &made.next;
        }
        let mut shown = f.debug_struct("ListCell");
        shown.field("values", &under.len()).field("under", &under);
        shown.finish()
    }
}

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
        // The ids a program states are one id space: asked of this list,
        // once a fill, in a build with debug assertions. A release build
        // writes the records of what the program stated.
        debug_assert!(
            lent.facts.is_none_or(ids_stand),
            "a list's facts state an id for every worker or for none, and no id twice"
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

/// Whether the ids that one list's facts state stand: one for every worker
/// of the list or for none, and no id twice. A zero states none.
/// [`FixedFacts::id`] asks the first of all a program's lists. A fill sees
/// one list, so a program that states ids on some of its lists passes it.
fn ids_stand(facts: &[FixedFacts]) -> bool {
    let stated = |fact: &FixedFacts| fact.id.filter(|id| *id != WorkerId(0));
    let mut ids: Vec<WorkerId> = facts.iter().filter_map(stated).collect();
    let count = ids.len();
    ids.sort_unstable();
    ids.dedup();
    ids.len() == count && (count == 0 || count == facts.len())
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
    /// An id the program states of a worker ([`FixedFacts::id`]) rides
    /// them: the list's records carry it at the worker's place. A program
    /// states one for every worker of its lists or for none, and no id
    /// twice in a list. A build with debug assertions checks each list on
    /// its own, where its value is made, and refuses one lent otherwise; a
    /// release build does not look.
    ///
    /// Facts of another count than the list are an error of the program,
    /// refused where the list's value is made, in every build, and the cell
    /// takes no value, so every later ask of the list lent so is refused
    /// again. The refusal is a panic. A pick passes it to its caller
    /// ([`InProcess::pick`]), and so does a pair's or an encode
    /// request's own `decide`, which a host calls itself. A decision
    /// through one of the engine's entries is answered
    /// `DecisionRefused::Faulted` and counted with the engine's faults: the
    /// entry catches the panic. The engine's task holds it and counts it
    /// with its own ([`InProcess::upkeep_faults`]).
    pub facts: Option<&'a [FixedFacts]>,
}

/// A worker list as a program publishes it: the list, the cell beside it,
/// and what the program states of the list's workers. It is what a program
/// keeps of each list it lends. [`Self::lent`] lends the three together,
/// so the cell goes out with no other list than its own.
///
/// A program builds one where it has the list at hand. [`Self::new`] states
/// nothing, reads no worker and takes no lock: it fits where a registry
/// makes the program's value of a generation under its own guard. A list
/// with facts is built with its workers read, so behind that guard: at the
/// list's first lending, in a once-cell of the program's value. Its facts
/// are whole from then on, and a lending reads that one cell and no second
/// one in here.
#[derive(Debug)]
pub struct PublishedList {
    workers: Arc<[Arc<dyn Worker>]>,
    facts: Option<Box<[FixedFacts]>>,
    cell: ListCell,
}

impl PublishedList {
    /// `workers` published with nothing stated of them: each states the
    /// default. Reads no worker and takes no lock.
    #[must_use]
    pub fn new(workers: Arc<[Arc<dyn Worker>]>) -> Self {
        Self {
            workers,
            facts: None,
            cell: ListCell::default(),
        }
    }

    /// `workers` published with what `facts_of` states of each: asked once
    /// a worker, in the list's order.
    #[must_use]
    pub fn with_facts(
        workers: Arc<[Arc<dyn Worker>]>,
        facts_of: impl FnMut(&Arc<dyn Worker>) -> FixedFacts,
    ) -> Self {
        let facts = workers.iter().map(facts_of).collect();
        Self::stating(workers, facts)
    }

    /// `workers` published with `facts`, which the program made in a pass
    /// of its own: one a worker, in the list's order.
    ///
    /// # Panics
    ///
    /// Facts of another count than the list are an error of the program,
    /// refused here in every build: a list lent so would be refused at
    /// each ask of it ([`LentList::facts`]).
    #[must_use]
    pub fn stating(workers: Arc<[Arc<dyn Worker>]>, facts: Box<[FixedFacts]>) -> Self {
        assert_eq!(
            facts.len(),
            workers.len(),
            "a list is published with one fact a worker"
        );
        Self {
            workers,
            facts: Some(facts),
            cell: ListCell::default(),
        }
    }

    /// The list, lent with its cell and its facts. A request lends each of
    /// its lists once, so this is read in the caller's own code.
    #[inline]
    #[must_use]
    pub fn lent(&self) -> LentList<'_> {
        LentList {
            workers: &self.workers,
            cell: &self.cell,
            facts: self.facts.as_deref(),
        }
    }

    /// The list.
    #[inline]
    #[must_use]
    pub fn workers(&self) -> &Arc<[Arc<dyn Worker>]> {
        &self.workers
    }
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
    /// ([`InProcess::set_lists`]): `hand` asks each list's value under
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

/// The lists a program handed the engine ([`InProcess::set_lists`]).
#[derive(Debug, Default)]
pub(super) struct HandedLists(OnceLock<Arc<dyn ModelLists>>);

impl HandedLists {
    /// The lists, where the program handed any: the engine's task asks
    /// for a word's models only then.
    pub(super) fn get(&self) -> Option<&Arc<dyn ModelLists>> {
        self.0.get()
    }
}

impl InProcess {
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
    /// Where the cell holds none under that document, the value is
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
    /// ([`InProcess::set_lists`]). Else the request that asks first
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
    /// ([`InProcess::upkeep_faults`]). The check stands in the fill and
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
            // A value made under another document is of the same list:
            // asked of each one the walk passes, so a cell lent with
            // another list is refused whichever engine asks it.
            debug_assert!(
                Arc::ptr_eq(made.members.workers(), workers),
                "a cell is lent with the list it stands beside"
            );
            cell = &made.next;
        }
    }
}

impl InProcess {
    /// [`Loaded::members_of`] under the document in force at the call: for
    /// a caller that asks a list's value outside a decision of the engine,
    /// as the engine's own task does ahead of a list's requests. Such a
    /// caller reads one document here and at the decision it asks beside
    /// this call only where no document is put in force between the two. A
    /// controller's deciding thread asks the value under its world's
    /// document, which its decision runs under ([`Loaded::members_of`]).
    #[inline]
    pub(crate) fn members_of<'a>(&self, lent: LentList<'a>) -> &'a PoolMembers {
        let document = self.in_force();
        self.under(&document).members_of(lent)
    }

    /// The bench's count: how many times the reports of `lent`'s list were
    /// converted under the document in force so far. It makes no value: a
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
}
