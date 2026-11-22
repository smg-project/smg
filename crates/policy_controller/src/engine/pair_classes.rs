//! The pair classes of two lists of workers, numbered from each prefill's
//! partners.
//!
//! A pair decision reads a class on every record and a table that says which
//! classes pair ([`PairTable`]). A caller that already knows which decodes
//! each prefill pairs with hands those lists here, once a membership, and
//! gets the class of every worker of both lists with the table. Nothing here
//! judges a pair: which workers pair is the caller's word.

use rustc_hash::FxHashMap;

use super::framework::PairTable;

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
        // Prefills that name the same decodes are one class.
        let mut by_list: FxHashMap<&'p [u32], u32> = FxHashMap::default();
        let mut lists: Vec<&'p [u32]> = Vec::new();
        let prefills: Box<[u32]> = partners
            .into_iter()
            .map(|list| {
                if !list.iter().any(|&place| (place as usize) < decodes) {
                    return 0;
                }
                *by_list.entry(list).or_insert_with(|| {
                    lists.push(list);
                    lists.len() as u32
                })
            })
            .collect();
        // What tells two decodes apart is the prefill classes that name
        // them. Each decode's are written one after another, ascending:
        // the classes are walked in their order.
        let held = |list: &'p [u32]| list.iter().map(|&place| place as usize);
        let mut starts = vec![0usize; decodes + 1];
        for place in lists.iter().flat_map(|list| held(list)) {
            if let Some(count) = starts.get_mut(place + 1).filter(|_| place < decodes) {
                *count += 1;
            }
        }
        for at in 0..decodes {
            starts[at + 1] += starts[at];
        }
        let mut named_by = vec![0u32; starts[decodes]];
        let mut filled = starts.clone();
        for (class, list) in (1u32..).zip(&lists) {
            for place in held(list).filter(|&place| place < decodes) {
                named_by[filled[place]] = class;
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
        for list in &lists {
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
        Self {
            prefills,
            decodes: decode_classes,
            ends: ends.into(),
            listed: listed.into(),
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
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Whether the classes pair prefill `prefill` with decode `decode`.
    fn pairs(classes: &PartnerClasses, prefill: usize, decode: usize) -> bool {
        classes.pairs(prefill, decode)
    }

    /// The classes of `partners` say the pairs `partners` names, and no
    /// other, under a table the engine honours whole.
    fn says(decodes: usize, partners: &[&[u32]]) -> PartnerClasses {
        let classes = PartnerClasses::number(decodes, partners.iter().copied());
        assert_eq!(classes.table().validate(), Ok(()));
        assert_eq!(classes.prefills().len(), partners.len());
        assert_eq!(classes.decodes().len(), decodes);
        for (prefill, list) in partners.iter().enumerate() {
            for decode in 0..decodes {
                assert_eq!(
                    pairs(&classes, prefill, decode),
                    list.contains(&(decode as u32)),
                    "prefill {prefill} with decode {decode}"
                );
            }
        }
        classes
    }

    #[test]
    fn a_fleet_whose_workers_all_pair_is_one_class_a_side() {
        let every: &[u32] = &[0, 1, 2];
        let classes = says(3, &[every, every]);
        assert_eq!(classes.prefills(), [1, 1]);
        assert_eq!(classes.decodes(), [1, 1, 1]);
        assert_eq!(classes.table().of(&1), [1]);
    }

    #[test]
    fn two_groups_that_never_pair_are_two_classes_a_side() {
        let classes = says(4, &[&[0, 1], &[2, 3], &[0, 1]]);
        assert_eq!(classes.prefills(), [1, 2, 1]);
        assert_eq!(classes.decodes(), [1, 1, 2, 2]);
        assert_eq!(classes.table().of(&1), [1]);
        assert_eq!(classes.table().of(&2), [2]);
    }

    /// A decode two prefill classes name is a class of its own, and both
    /// lists name it: a worker stands once in its list.
    #[test]
    fn a_decode_two_prefill_classes_share_is_named_by_both_lists() {
        let classes = says(3, &[&[0, 1], &[1, 2]]);
        assert_eq!(classes.prefills(), [1, 2]);
        assert_eq!(classes.decodes(), [1, 2, 3]);
        assert_eq!(classes.table().of(&1), [1, 2]);
        assert_eq!(classes.table().of(&2), [2, 3]);
    }

    #[test]
    fn a_worker_nobody_pairs_with_is_class_zero() {
        let classes = says(3, &[&[], &[2]]);
        assert_eq!(classes.prefills(), [0, 1]);
        assert_eq!(classes.decodes(), [0, 0, 1]);
    }

    #[test]
    fn a_place_past_the_decode_list_names_no_decode() {
        let classes = PartnerClasses::number(2, [&[1u32, 7][..], &[9][..]]);
        assert_eq!(classes.table().validate(), Ok(()));
        assert_eq!(classes.prefills(), [1, 0]);
        assert_eq!(classes.decodes(), [0, 1]);
        assert!(pairs(&classes, 0, 1));
    }

    /// No prefill and no decode: no class, and no table. A decision over
    /// such classes pairs nothing, since class `0` pairs with nobody.
    #[test]
    fn empty_lists_have_no_class_and_no_table() {
        let classes = says(0, &[]);
        assert!(classes.table().is_empty());
        let unpaired = says(2, &[&[], &[]]);
        assert_eq!(unpaired.prefills(), [0, 0]);
        assert_eq!(unpaired.decodes(), [0, 0]);
        assert!(unpaired.table().is_empty());
    }

    /// A relation with no pattern to it: every prefill's list is kept.
    #[test]
    fn the_classes_say_the_pairs_they_were_numbered_from() {
        let lists: Vec<Vec<u32>> = (0u32..12)
            .map(|prefill| {
                (0u32..9)
                    .filter(|decode| (prefill * 7 + decode * 5 + prefill * decode) % 3 == 0)
                    .collect()
            })
            .collect();
        let partners: Vec<&[u32]> = lists.iter().map(Vec::as_slice).collect();
        says(9, &partners);
    }
}
