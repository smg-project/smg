//! The pair classes of two lists of workers, numbered from each prefill's
//! partners.
//!
//! A pair decision reads a class on every record and a table that says which
//! classes pair ([`PairTable`]). A caller that already knows which decodes
//! each prefill pairs with hands those lists here, once a membership, and
//! gets the class of every worker of both lists with the table. Nothing here
//! judges a pair: which workers pair is the caller's word.

use rustc_hash::FxHashMap;

use super::framework::{FirstPartners, PairTable};

/// The class of every worker of a prefill list and of a decode list, and
/// the table of the classes that pair.
///
/// Prefills that pair with the same decodes share a class, and so do decodes
/// that pair with the same prefills. Each side's classes are numbered from
/// `1` in the order their first worker stands in its list. A worker that
/// pairs with nobody is class `0`, which is outside any pairing.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct PartnerClasses {
    /// Each prefill's class, in the prefill list's order.
    prefills: Box<[u32]>,
    /// Each decode's class, in the decode list's order.
    decodes: Box<[u32]>,
    /// Where prefill class `k`'s list ends in `listed`, at `k - 1`.
    ends: Box<[u32]>,
    /// The lists one after another: the decode classes each prefill class
    /// pairs with, ascending.
    listed: Box<[u32]>,
    /// Where prefill class `k`'s first partners end in `first_listed`, at
    /// `k - 1`. Empty where no prefill has a first partner.
    first_ends: Box<[u32]>,
    /// The first partners' lists one after another: the decode classes each
    /// prefill class is paired with first, ascending.
    first_listed: Box<[u32]>,
}

impl PartnerClasses {
    /// The classes of a prefill list and a decode list of `decodes` workers.
    ///
    /// `partners` yields one list for each prefill, in the prefill list's
    /// order: the places in the decode list of the decodes that prefill
    /// pairs with, ascending, each place once. A place past the decode list
    /// names no decode and is passed over.
    ///
    /// The table numbers its decode classes from `1` with none left out,
    /// and each is named by some prefill class's list, so it is a table the
    /// engine honours whole ([`PairTable::validate`]).
    ///
    /// Called where a membership changes, never at a request: it allocates,
    /// and its work grows with the pairs.
    pub fn number<'p>(decodes: usize, partners: impl IntoIterator<Item = &'p [u32]>) -> Self {
        Self::number_first(decodes, partners.into_iter().map(|list| (list, &[][..])))
    }

    /// The classes of [`Self::number`] where some partners go first
    /// ([`FirstPartners`]). `partners` yields two lists for each prefill:
    /// its partners as [`Self::number`] takes them, and the places of those
    /// of them it is paired with first, ascending. A place the first list
    /// names outside the partners is passed over.
    ///
    /// Prefills share a class where both their lists are the same, and
    /// decodes where the same prefill classes name them and the same of
    /// those name them first: a class's first partners are then whole
    /// classes of its table, and they stand within it
    /// ([`FirstPartners::within`]).
    ///
    /// Where every prefill is paired first with all its partners, no pair
    /// goes ahead of another, and no first partner is kept: a decision
    /// then reads none ([`Self::first`]).
    pub fn number_first<'p>(
        decodes: usize,
        partners: impl IntoIterator<Item = (&'p [u32], &'p [u32])>,
    ) -> Self {
        // Prefills that name the same decodes, and the same of them first,
        // are one class.
        let mut by_list: FxHashMap<(&'p [u32], &'p [u32]), u32> = FxHashMap::default();
        let mut lists: Vec<(&'p [u32], &'p [u32])> = Vec::new();
        let prefills: Box<[u32]> = partners
            .into_iter()
            .map(|both| {
                if !both.0.iter().any(|&place| (place as usize) < decodes) {
                    return 0;
                }
                *by_list.entry(both).or_insert_with(|| {
                    lists.push(both);
                    lists.len() as u32
                })
            })
            .collect();
        // What tells two decodes apart is the prefill classes that name
        // them. Each decode's are written one after another, ascending:
        // the classes are walked in their order.
        let held = |list: &'p [u32]| list.iter().map(|&place| place as usize);
        let mut starts = vec![0usize; decodes + 1];
        for place in lists.iter().flat_map(|both| held(both.0)) {
            if let Some(count) = starts.get_mut(place + 1).filter(|_| place < decodes) {
                *count += 1;
            }
        }
        for at in 0..decodes {
            starts[at + 1] += starts[at];
        }
        let mut named_by = vec![0u32; starts[decodes]];
        let mut filled = starts.clone();
        // A name says its class and whether that class names the decode
        // first: decodes one class takes first stand apart from its others.
        for (class, (list, first)) in (1u32..).zip(&lists) {
            for place in held(list).filter(|&place| place < decodes) {
                let goes_first = first.binary_search(&(place as u32)).is_ok();
                named_by[filled[place]] = class << 1 | u32::from(goes_first);
                filled[place] += 1;
            }
        }
        // Decodes named by the same prefill classes are one class.
        let mut by_names: FxHashMap<&[u32], u32> = FxHashMap::default();
        let decode_classes: Box<[u32]> = (0..decodes)
            .map(|at| {
                let names = &named_by[starts[at]..starts[at + 1]];
                if names.is_empty() {
                    return 0;
                }
                let next = by_names.len() as u32 + 1;
                *by_names.entry(names).or_insert(next)
            })
            .collect();
        // The table: for each prefill class, the classes of its decodes.
        let mut ends = Vec::with_capacity(lists.len());
        let mut listed: Vec<u32> = Vec::new();
        for (list, _) in &lists {
            let from = listed.len();
            listed.extend(
                held(list)
                    .filter(|&place| place < decodes)
                    .map(|place| decode_classes[place]),
            );
            listed[from..].sort_unstable();
            let mut kept = from;
            for at in from..listed.len() {
                if at == from || listed[at] != listed[kept - 1] {
                    listed[kept] = listed[at];
                    kept += 1;
                }
            }
            listed.truncate(kept);
            ends.push(listed.len() as u32);
        }
        // The first partners, where a prefill has one: for each prefill
        // class, the classes of the partners it names first, ascending.
        let (mut first_ends, mut first_listed) = (Vec::new(), Vec::new());
        if lists.iter().any(|(_, first)| !first.is_empty()) {
            for (list, first) in &lists {
                let partner = |place: &usize| list.binary_search(&(*place as u32)).is_ok();
                let own = held(first).filter(|place| *place < decodes && partner(place));
                let mut classes: Vec<u32> = own.map(|place| decode_classes[place]).collect();
                classes.sort_unstable();
                classes.dedup();
                first_listed.extend(classes);
                first_ends.push(first_listed.len() as u32);
            }
            // Every partner first is no partner first.
            if first_listed == listed {
                first_ends.clear();
                first_listed.clear();
            }
        }
        Self {
            prefills,
            decodes: decode_classes,
            ends: ends.into(),
            listed: listed.into(),
            first_ends: first_ends.into(),
            first_listed: first_listed.into(),
        }
    }

    /// The classes as a pairing rule derived them, with its table: each
    /// prefill's class and each decode's in their lists' order, where
    /// prefill class `k`'s list ends in `listed` (at `k - 1`), and the
    /// lists one after another, the decode classes each prefill class
    /// pairs with, ascending. `first` is the rule's first partners, laid
    /// out as the table is: their ends, then their lists. Both are empty
    /// where the rule puts no pair ahead of another.
    ///
    /// For a rule that numbers as [`Self::number_first`] does: each side's
    /// classes from `1`, `0` for a worker that pairs with nobody, every
    /// decode class named by some list, and a class's first partners whole
    /// classes of its list.
    #[must_use]
    pub(crate) fn derived(
        prefills: Box<[u32]>,
        decodes: Box<[u32]>,
        [ends, listed]: [Box<[u32]>; 2],
        [first_ends, first_listed]: [Box<[u32]>; 2],
    ) -> Self {
        Self {
            prefills,
            decodes,
            ends,
            listed,
            first_ends,
            first_listed,
        }
    }

    /// Each prefill's class, in the prefill list's order: what a caller
    /// writes into that list's records
    /// ([`Candidate::pair_class`](super::view::Candidate::pair_class)).
    #[must_use]
    pub fn prefills(&self) -> &[u32] {
        &self.prefills
    }

    /// Each decode's class, in the decode list's order.
    #[must_use]
    pub fn decodes(&self) -> &[u32] {
        &self.decodes
    }

    /// Which classes pair, as a pair decision reads it beside its records
    /// ([`PairPools::table`](super::framework::PairPools::table)).
    #[must_use]
    pub fn table(&self) -> PairTable<'_> {
        PairTable::new(&self.ends, &self.listed)
    }

    /// The partners each prefill class is paired with first, as a pair
    /// decision reads them beside the table
    /// ([`PairPools::first`](super::framework::PairPools::first)): none
    /// where the classes were numbered with no first partner.
    #[must_use]
    pub fn first(&self) -> FirstPartners<'_> {
        FirstPartners::new(&self.first_ends, &self.first_listed)
    }

    /// Whether the classes pair the prefill at place `prefill` of its list
    /// with the decode at place `decode` of its own. `false` for a place
    /// past either list.
    #[must_use]
    pub fn pairs(&self, prefill: usize, decode: usize) -> bool {
        let (Some(class), Some(&partner)) = (self.prefills.get(prefill), self.decodes.get(decode))
        else {
            return false;
        };
        partner != 0 && self.table().of(class).contains(&partner)
    }

    /// Whether the classes pair that prefill with that decode first
    /// ([`Self::first`]). `false` for a place past either list, and for
    /// every pair where no partner goes first.
    #[must_use]
    pub fn pairs_first(&self, prefill: usize, decode: usize) -> bool {
        let (Some(&class), Some(&partner)) = (self.prefills.get(prefill), self.decodes.get(decode))
        else {
            return false;
        };
        partner != 0 && self.first().of(class).contains(&partner)
    }
}

#[cfg(test)]
#[path = "../../tests/unit/engine/pair_classes/tests.rs"]
mod tests;
