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
#[non_exhaustive]
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

impl<'a> LentList<'a> {
    /// A list and the cell its program keeps beside it, with no facts of its
    /// workers: every worker then states the default ([`Self::facts`]). A
    /// program that knows its workers assigns them. It checks nothing. Code
    /// outside this crate starts here: the record gains fields, so no literal
    /// of it builds there.
    #[inline]
    #[must_use]
    pub const fn of(workers: &'a Arc<[Arc<dyn Worker>]>, cell: &'a ListCell) -> Self {
        Self {
            workers,
            cell,
            facts: None,
        }
    }
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
#[path = "../../tests/unit/engine/list_cell/tests.rs"]
pub(crate) mod tests;
