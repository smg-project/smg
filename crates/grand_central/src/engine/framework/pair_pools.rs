//! A pair's two pools, the table that says which of their classes pair,
//! and the partners a class is paired with first.
//!
//! A pair's decision is handed its two pools in one value ([`PairPools`]):
//! the prefills and the decodes, beside each pool its affinity column
//! and its signal cells, the pair table, the first partners, and whether
//! the pools are a part of the lists the request's caller lent. The pair
//! table ([`PairTable`]) names for each prefill class the decode classes
//! it pairs with, and the first partners ([`FirstPartners`]) name the part
//! of them it is paired with first. A host builds both over its own
//! slices. It can ask once whether its table is one the engine honours
//! whole ([`PairTable::validate`]) and whether its first partners stand
//! within that table ([`FirstPartners::within`]): [`PairTableError`] and
//! [`FirstPartnersError`] name the clause that fails.
//!
//! This module is a child of the framework's because it reads private
//! items of that module as they stand: [`ClassBits`] and its methods
//! `reset`, `insert`, `has` and `classes`, [`column_fits`],
//! [`debug_assert_indexed`] and [`debug_assert_cells`].

// `Engine` is named only by doc links of this module.
#[cfg(doc)]
use super::Engine;
use super::{column_fits, debug_assert_cells, debug_assert_indexed, ClassBits};
use crate::{
    engine::{
        decision::PairSide,
        view::{AffinityFacts, Candidate, Phase, RequestFacts},
    },
    load::signals::SignalCells,
    Draw,
};

/// The two pools of a pair decision: the open prefills and the decodes, each
/// row carrying its pair class; the pair table, which says which classes
/// pair; and beside each pool its affinity column, by the candidate's index,
/// empty for a side nothing reads it on, and its signal cells, none for a
/// side whose stages read no signal.
#[derive(Clone, Copy, Default)]
#[non_exhaustive]
pub struct PairPools<'a> {
    pub prefills: &'a [Candidate],
    pub decodes: &'a [Candidate],
    /// The prefills' affinity facts: what the prefill side's stages find in
    /// [`RequestFacts::affinity`].
    pub prefill_affinity: &'a [AffinityFacts],
    /// The decodes' affinity facts, the same for the decode side's stages.
    pub decode_affinity: &'a [AffinityFacts],
    /// The prefills' signal cells: what the prefill side's stages find in
    /// [`RequestFacts::signal_cells`].
    pub prefill_signal_cells: Option<&'a SignalCells>,
    /// The decodes' signal cells, the same for the decode side's stages.
    pub decode_signal_cells: Option<&'a SignalCells>,
    /// The decode classes each prefill class pairs with. Left at its
    /// default, classes pair by their number.
    pub table: PairTable<'a>,
    /// The partners each prefill class is paired with first. Left at its
    /// default, no partner goes first.
    pub first: FirstPartners<'a>,
    /// Whether the pools are a part of the lists the request's caller
    /// lent. Two things make a part. A hard demand its door judged left
    /// each side the workers that meet it. Or the encode pass is laid for
    /// the request, which states an item or holds its decode to the ones
    /// that take images: the pass hands the pair the workers its
    /// encoders and its images leave. Three writers say so, each from
    /// the test it already makes: the pair's door and the encode door
    /// where the request states a demand, and the encode pass where it
    /// is laid.
    ///
    /// Nothing else makes a part. The places a host states for a side are
    /// its own word of its list. The workers a route's rule takes are the
    /// call's places for its later decisions. Under both the pools are
    /// whole.
    ///
    /// Behind a part a pair's decision does not ask its one question of a
    /// shed ([`Engine::decide_pair`]). Private to the crate: a host of its
    /// own builds the pools from their default, which says whole.
    pub(crate) part: bool,
}

/// Which pair classes pair: for each prefill class, the decode classes it
/// pairs with. A worker stands once in its pool and carries one class
/// ([`Candidate::pair_class`]): prefills that pair with the same decodes
/// share one, and so do decodes that pair with the same prefills. The table
/// names classes, never rows or positions, so one table serves every pool
/// narrowed from the same workers.
///
/// The default is no table: a prefill class then pairs with the decode
/// class of its own number, as under a host that gives a decode the class
/// of the prefills it partners.
///
/// A host builds a table ([`PairTable::new`]) and can ask once whether it
/// is one the engine honours whole ([`PairTable::validate`]). How a decision
/// reads it is the engine's own.
#[derive(Clone, Copy, Debug, Default)]
pub struct PairTable<'a> {
    /// Where each prefill class's list ends in `classes`.
    ends: &'a [u32],
    /// The lists, one after another.
    classes: &'a [u32],
}

impl<'a> PairTable<'a> {
    /// A table over the host's two slices. `classes` holds the lists one
    /// after another, prefill class 1's first, and `ends[k - 1]` is where
    /// prefill class `k`'s list ends in it. A list names decode classes in
    /// ascending order.
    ///
    /// A host numbers its decode classes densely from 1, each named by some
    /// list, so a list's classes are the numbers from 1 to the count of the
    /// lists' entries. What a list names outside them is no class of the
    /// table, and no decode pairs through it: class 0, which is outside any
    /// pairing, and a number above the count. A decision keeps one bit a
    /// class number up to that count, never more.
    ///
    /// Nothing is checked here, so a decision of a release build pays for no
    /// check: a table that fails [`PairTable::validate`] costs it the pairs the
    /// table misnames and a set of at most the table's own size, never the
    /// process.
    pub const fn new(ends: &'a [u32], classes: &'a [u32]) -> Self {
        Self { ends, classes }
    }

    /// No table was handed: classes pair by their number.
    pub(crate) const fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }

    /// The decode classes prefill class `class` pairs with, ascending: its
    /// list under a table, its own number under none. Every read of a pair
    /// decision asks here, but for the width's walk: it reads each list's
    /// last entry by the same cut ([`Self::cut`]), and asks here list by
    /// list once a last entry stands above the count. Class 0 pairs with
    /// nothing, as does a class the table holds no list for, and a list is
    /// read up to the count of the lists' entries: a number above it is no
    /// class of the table. The class is lent, so that under no table the
    /// list is the number itself.
    pub(crate) fn of<'c>(&self, class: &'c u32) -> &'c [u32]
    where
        'a: 'c,
    {
        if *class == 0 {
            return &[];
        }
        if self.is_empty() {
            return std::slice::from_ref(class);
        }
        let list = self.list(*class as usize - 1);
        // A list ascends, so the numbers above the count are its end.
        let held = |listed: &u32| *listed as usize <= self.classes.len();
        match list.last() {
            Some(last) if !held(last) => &list[..list.partition_point(held)],
            _ => list,
        }
    }

    /// The list of the prefill class at `at`, as the host handed it; empty
    /// for a class the table holds no list for.
    fn list(&self, at: usize) -> &'a [u32] {
        let Some(&end) = self.ends.get(at) else {
            return &[];
        };
        let start = at.checked_sub(1).map_or(0, |before| self.ends[before]);
        self.cut(start as usize, end as usize)
    }

    /// The entries from `start` up to `end`: the one cut of a list out of
    /// the lists' entries, empty where the two ends do not hold.
    /// [`Self::list`] cuts by a class's own two ends, and the width's walk
    /// by the end it passed last.
    fn cut(&self, start: usize, end: usize) -> &'a [u32] {
        self.classes.get(start..end).unwrap_or(&[])
    }

    /// The table's prefill classes.
    fn prefill_classes(&self) -> std::ops::RangeInclusive<u32> {
        1..=self.ends.len() as u32
    }

    /// The width of a set of bits over the table's decode classes: one past
    /// the largest class a list names, which is at most the count of the
    /// lists' entries ([`PairTable::of`]). A table costs a decision a set
    /// of its own size, whatever numbers it names. Lists ascend, so a
    /// list's last entry is its largest; the width is capped at the count
    /// all the same, so the bound stands for a list out of order too.
    ///
    /// One walk of the lists' ends reads each list's last entry by the cut
    /// a list's own read makes ([`Self::cut`]). The walk stays out of line
    /// and calls nothing on its way, so the routine that asks is compiled
    /// as it was without a table's walk in it. The first last entry above
    /// the count ends the walk: [`Self::width_by_lists`] answers.
    #[inline(never)]
    pub(super) fn width(&self) -> usize {
        let count = self.classes.len();
        let (mut largest, mut start) = (0u32, 0usize);
        for &end in self.ends {
            let end = end as usize;
            if let Some(&last) = self.cut(start, end).last() {
                if last as usize > count {
                    return self.width_by_lists();
                }
                largest = largest.max(last);
            }
            start = end;
        }
        Self::capped(largest, count)
    }

    /// The width of a table with a list's last entry above the count of
    /// the entries: each list's own read says where it ends
    /// ([`PairTable::of`]), list by list. Cold and out of line: a table a
    /// host numbers from one up never comes here.
    #[cold]
    #[inline(never)]
    pub(super) fn width_by_lists(&self) -> usize {
        let largest = self
            .prefill_classes()
            .filter_map(|class| self.of(&class).last().copied())
            .max()
            .unwrap_or(0);
        Self::capped(largest, self.classes.len())
    }

    /// One past `largest`, and at most one past `count`: the width's cap,
    /// apart from the lists so that a case asks it. A list's read ends at
    /// the count ([`PairTable::of`]), so no table hands the width a larger
    /// number while the library's search answers as it does today.
    pub(super) fn capped(largest: u32, count: usize) -> usize {
        (largest as usize).min(count) + 1
    }

    /// Whether this is a table the engine honours whole, and the first
    /// clause it fails when it is not: the ends ascend and the last stands
    /// where the lists end, and each list names its classes in ascending
    /// order, none of them class 0 and none above the count of the lists'
    /// entries. The default, no table, is valid.
    ///
    /// One question, two askers: a host asks it once where it builds a
    /// table, in every build, and a debug build asks it where a decision
    /// takes one.
    pub fn validate(&self) -> Result<(), PairTableError> {
        self.laid_out(true)
    }

    /// The clauses of [`Self::validate`], the last of them only where
    /// `counted`: lists that name some of a table's classes hold numbers
    /// above their own count ([`FirstPartners::within`]).
    fn laid_out(&self, counted: bool) -> Result<(), PairTableError> {
        if !self.ends.windows(2).all(|pair| pair[0] <= pair[1]) {
            return Err(PairTableError::EndsDescend);
        }
        // Every entry stands in a list, so the count a read holds a list
        // to is the lists' own.
        let entries = self.classes.len();
        let end = self.ends.last().map_or(0, |&end| end as usize);
        if end > entries {
            return Err(PairTableError::EndPastLists);
        }
        if end < entries {
            return Err(PairTableError::EntriesPastEnds);
        }
        for list in (0..self.ends.len()).map(|at| self.list(at)) {
            if !list.windows(2).all(|pair| pair[0] < pair[1]) {
                return Err(PairTableError::ListOutOfOrder);
            }
            if list.first() == Some(&0) {
                return Err(PairTableError::ClassZero);
            }
            if counted && list.last().is_some_and(|&last| last as usize > entries) {
                return Err(PairTableError::NumberAboveCount);
            }
        }
        Ok(())
    }

    /// The prefill classes with a partner, ascending, into `classes`. `open`
    /// is the class of each decode the decode profile's hard filters keep.
    /// Under a table the question is asked once a prefill class, each
    /// stopping at the first open class it lists, and of no class when no
    /// decode is open. Under none the open classes are the answer: a class
    /// pairs with its own number.
    pub(super) fn partnered(
        &self,
        open: impl Iterator<Item = u32>,
        classes: &mut Vec<u32>,
        marks: &mut ClassBits,
    ) {
        classes.clear();
        if self.is_empty() {
            classes.extend(open);
            classes.sort_unstable();
            classes.dedup();
            return;
        }
        // A number above the table's width is no class of the table.
        let width = self.width();
        marks.reset(width);
        // The decodes are read until every number from one to the width
        // less one has an open decode: no decode behind that one opens a
        // class, and `open` is not asked for it.
        let mut closed = width.saturating_sub(1);
        let mut any = false;
        for class in open {
            if class == 0 || class as usize >= width || marks.has(class) {
                continue;
            }
            marks.insert(class);
            any = true;
            closed -= 1;
            if closed == 0 {
                break;
            }
        }
        if any {
            classes.extend(
                self.prefill_classes()
                    .filter(|class| self.of(class).iter().any(|&listed| marks.has(listed))),
            );
        }
    }

    /// Turns the prefill classes in `classes` into the decode classes they
    /// pair with, both ascending. Under no table nothing moves: a class
    /// pairs with its own number.
    pub(super) fn listed(&self, classes: &mut Vec<u32>, marks: &mut ClassBits) {
        if self.is_empty() {
            return;
        }
        marks.reset(self.width());
        for class in classes.iter() {
            for &listed in self.of(class) {
                marks.insert(listed);
            }
        }
        classes.clear();
        classes.extend(marks.classes());
    }
}

/// The clause of [`PairTable::validate`] a host's table fails.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum PairTableError {
    #[error("an end stands before the end of the list before it")]
    EndsDescend,
    #[error("an end stands past the lists")]
    EndPastLists,
    #[error("the lists hold entries past the last end")]
    EntriesPastEnds,
    #[error("a list names its classes out of ascending order")]
    ListOutOfOrder,
    #[error("a list names class 0")]
    ClassZero,
    #[error("a list names a number above the count of the lists' entries")]
    NumberAboveCount,
}

/// The partners each prefill class is paired with first: for each prefill
/// class, a part of the decode classes its list in the pair table names
/// ([`PairTable`]). While one of them is open, the pair is made among the
/// prefills that have one open and their first partners, and a request
/// waits where such a prefill is held back by its admission limit alone.
/// Every other pair is made only while no prefill has a first partner
/// open ([`Engine::decide_pair`]). The default is none: every partner
/// stands alike.
///
/// The lists name classes of the pair table and not all of them, so they
/// are no table of their own. A host asks once, where it builds the two,
/// whether they stand within its table ([`FirstPartners::within`]).
#[derive(Clone, Copy, Debug, Default)]
pub struct FirstPartners<'a> {
    /// The lists in a table's storage, read as the host handed them.
    lists: PairTable<'a>,
}

impl<'a> FirstPartners<'a> {
    /// The first partners over the host's two slices, laid out as a pair
    /// table's are ([`PairTable::new`]): `classes` holds the lists one
    /// after another, prefill class 1's first, and `ends[k - 1]` is where
    /// prefill class `k`'s list ends in it. A list names decode classes in
    /// ascending order. Nothing is checked here.
    pub const fn new(ends: &'a [u32], classes: &'a [u32]) -> Self {
        Self {
            lists: PairTable::new(ends, classes),
        }
    }

    /// The first partners of prefill class `class`, as the host handed
    /// them: none for class 0 and for a class the lists do not reach.
    pub(crate) fn of(&self, class: u32) -> &'a [u32] {
        match class.checked_sub(1) {
            Some(at) => self.lists.list(at as usize),
            None => &[],
        }
    }

    /// The prefill classes with an open first partner, ascending, into
    /// `classes`, and whether there is one. `marks` holds the open decode
    /// classes as `table`'s own pass left them ([`PairTable::partnered`]).
    /// Under no table that pass marks nothing, and no first partner is
    /// read.
    pub(super) fn open(
        &self,
        table: &PairTable<'_>,
        marks: &ClassBits,
        classes: &mut Vec<u32>,
    ) -> bool {
        if table.is_empty() {
            return false;
        }
        classes.clear();
        let open = |class: &u32| self.of(*class).iter().any(|&listed| marks.has(listed));
        classes.extend(self.lists.prefill_classes().filter(open));
        !classes.is_empty()
    }

    /// Whether these are first partners of `table`, and the first clause
    /// they fail when they are not. The lists are laid out as a table's
    /// are, by the table's own clauses but for its count
    /// ([`PairTable::validate`]): the ends ascend and the last stands where
    /// the lists end, and each list names its classes in ascending order,
    /// none of them class 0. Then each class a list names is a partner of
    /// its prefill class in `table`. The default, none, stands within
    /// every table.
    ///
    /// A host asks it once where it builds the two, and a debug build asks
    /// it where a decision takes them.
    pub fn within(&self, table: &PairTable<'_>) -> Result<(), FirstPartnersError> {
        self.lists
            .laid_out(false)
            .map_err(FirstPartnersError::Lists)?;
        let partners = |class: u32| {
            let listed = table.of(&class);
            let partner = |named: &u32| listed.binary_search(named).is_ok();
            self.of(class).iter().all(partner)
        };
        if self.lists.prefill_classes().all(partners) {
            Ok(())
        } else {
            Err(FirstPartnersError::NoPartner)
        }
    }
}

/// The clause of [`FirstPartners::within`] a host's first partners fail.
#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum FirstPartnersError {
    #[error("the first partners' lists: {0}")]
    Lists(PairTableError),
    #[error("a first partner is no partner of its prefill class in the pair table")]
    NoPartner,
}

impl<'a> PairPools<'a> {
    /// The two pools with nothing beside them: no affinity column, no
    /// signal cells and no pair table, so classes pair by their number.
    /// A caller assigns what it holds beside its pools over it:
    /// `let mut pools = PairPools::of(prefills, decodes); pools.table = table;`.
    #[must_use]
    pub fn of(prefills: &'a [Candidate], decodes: &'a [Candidate]) -> Self {
        Self {
            prefills,
            decodes,
            ..Self::default()
        }
    }

    /// The decode classes a prefill of `class` is paired among: its first
    /// partners where the pair is made among those (`firsts`), else every
    /// class its list in the table names.
    pub(super) fn partners<'c>(&self, class: &'c u32, firsts: bool) -> &'c [u32]
    where
        'a: 'c,
    {
        if firsts {
            self.first.of(*class)
        } else {
            self.table.of(class)
        }
    }

    /// One side of the pair as its stages read it: the side's pool, and the
    /// request's facts with the side's phase, its draw sequence and its own
    /// pool's affinity column and signal cells, and with no rows named:
    /// those index neither pool. `None` when that column was not filled
    /// for that pool ([`column_fits`]). The one place a pair's column and
    /// cells meet their pool, so no call site hands one side's with the
    /// other side's candidates.
    ///
    /// A side reads its own places on a lent ring in the one list
    /// ([`RequestFacts::ring_positions`]): the decode side's own where the
    /// request states them apart
    /// ([`RequestFacts::decode_ring_positions`]), which neither side's
    /// facts carry on. So a stage is never shown the other side's list.
    pub(super) fn side(
        self,
        side: PairSide,
        facts: &RequestFacts<'a>,
    ) -> Option<(&'a [Candidate], RequestFacts<'a>)> {
        let (pool, affinity, signal_cells, phase, seed, ring_positions) = match side {
            PairSide::Prefill => (
                self.prefills,
                self.prefill_affinity,
                self.prefill_signal_cells,
                Phase::Prefill,
                facts.seed,
                facts.ring_positions,
            ),
            // The decode side draws its own sequence, derived from the
            // request's seed: one seed on both sides made the decode picker
            // replay the prefill picker's draws, so the decode chosen
            // followed the prefill chosen (random over four prefills of four
            // decodes each reached four of the sixteen). The decision still
            // replays from the trace's seed.
            PairSide::Decode => (
                self.decodes,
                self.decode_affinity,
                self.decode_signal_cells,
                Phase::Decode,
                Draw::new(facts.seed).next_u64(),
                facts.decode_ring_positions.unwrap_or(facts.ring_positions),
            ),
        };
        debug_assert_indexed(pool);
        debug_assert_cells(signal_cells, pool);
        column_fits(affinity, pool).then(|| {
            let facts = RequestFacts {
                phase,
                seed,
                affinity,
                named: None,
                // The side's own cells: the request's would be read by
                // another pool's positions.
                signal_cells,
                // The side's own places, in the one list a stage reads.
                ring_positions,
                decode_ring_positions: None,
                ..*facts
            };
            (pool, facts)
        })
    }
}
