//! The hash ring of one list of workers.
//!
//! A host that keeps its workers in lists of its own, as a gateway that
//! decides in its own process does, can keep one ring a list. The ring is
//! built from what the list's rows state for good, their ids and their
//! stable keys, and is never changed: the list a host publishes at a
//! membership change is a new list, and takes a new ring. So a reader takes
//! no lock: it holds the ring whole, behind one pointer.
//!
//! A member is named by its row's place in the list: a walk hands out places,
//! and its caller reads the row there with no search for it.

use std::fmt;

use super::{key_position, position, FREE, VIRTUAL_NODES};
use crate::engine::{decision::MAX_TARGETS, framework::DRAW_BOUND, view::WorkerId};

/// The entries a slice of the key space holds on average
/// ([`PoolRing::starts`]): a walk's first entry is found among that many.
const ENTRIES_A_SLICE: usize = 16;

/// The hash ring of one list of workers: each row at [`VIRTUAL_NODES`]
/// positions hashed from its stable key, in the ring's order.
///
/// The positions are the ones the scorer's own ring gives the same members
/// ([`Ring`]), and two members at one position stand in the order of their
/// ids, as they do there. So a key has one owner by either ring, and the
/// members behind the owner follow in one order.
///
/// Two rows of one id are one member, as in the scorer's own ring: the first
/// of them stands for both ([`Self::is_distinct`]).
///
/// [`Ring`]: super::Ring
pub struct PoolRing {
    /// `(position, row)` in the ring's order: by position, and at one
    /// position by the rows' ids.
    entries: Box<[(u64, u32)]>,
    /// Where each slice of the key space starts among the entries: the slice
    /// `s` holds the positions whose top bits are `s`, and its entries are
    /// `starts[s]..starts[s + 1]`. A walk finds its first entry in one
    /// slice, a few entries, where a search of every entry reads a line of
    /// memory a step.
    starts: Box<[u32]>,
    /// How far right a position is moved to name its slice.
    shift: u32,
    /// How many rows the list holds.
    rows: u32,
    /// Where two rows of the list carry one id: for each row, the first row
    /// of its id, which stands for it on the ring. Empty where every row
    /// has an id of its own.
    stands_for: Box<[u32]>,
}

impl PoolRing {
    /// The ring of a list whose rows carry these ids and stable keys, in the
    /// list's order: a row's place in `rows` is the place a walk names it
    /// by. A row whose id an earlier row carries joins under that row, with
    /// its key.
    ///
    /// Called once a list. It allocates, hashes [`VIRTUAL_NODES`] positions
    /// a row and sorts them.
    #[must_use]
    pub fn new(rows: impl IntoIterator<Item = (WorkerId, u32)>) -> Self {
        let rows: Vec<(WorkerId, u32)> = rows.into_iter().collect();
        // The rows by id, a row that repeats an id behind the first of it.
        let mut by_id: Vec<(WorkerId, u32)> =
            (0u32..).zip(&rows).map(|(at, row)| (row.0, at)).collect();
        by_id.sort_unstable();
        let mut stands_for: Vec<u32> = Vec::new();
        let mut entries: Vec<(u64, WorkerId, u32)> =
            Vec::with_capacity(rows.len() * VIRTUAL_NODES as usize);
        let mut first: Option<(WorkerId, u32)> = None;
        for &(id, at) in &by_id {
            match first {
                Some((standing, member)) if standing == id => {
                    if stands_for.is_empty() {
                        stands_for.extend(0..rows.len() as u32);
                    }
                    stands_for[at as usize] = member;
                }
                _ => {
                    first = Some((id, at));
                    let key = rows[at as usize].1;
                    entries.extend((0..VIRTUAL_NODES).map(|vnode| (position(key, vnode), id, at)));
                }
            }
        }
        // By position, and two members at one position by their ids: the
        // order the scorer's own ring keeps.
        entries.sort_unstable();
        let entries: Box<[(u64, u32)]> = entries
            .into_iter()
            .map(|(position, _, row)| (position, row))
            .collect();
        let slices = (entries.len() / ENTRIES_A_SLICE).next_power_of_two().max(2);
        let shift = u64::BITS - slices.trailing_zeros();
        let mut starts = vec![0u32; slices + 1];
        // The next slice to open: every slice up to an entry's own starts
        // at that entry, and the slices past the last entry at the end.
        let mut next = 0usize;
        for (at, &(position, _)) in entries.iter().enumerate() {
            let slice = (position >> shift) as usize;
            while next <= slice {
                starts[next] = at as u32;
                next += 1;
            }
        }
        for start in &mut starts[next..] {
            *start = entries.len() as u32;
        }
        Self {
            entries,
            starts: starts.into(),
            shift,
            rows: rows.len() as u32,
            stands_for: stands_for.into(),
        }
    }

    /// How many rows the list holds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rows as usize
    }

    /// Whether the list holds no row.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rows == 0
    }

    /// Whether every row of the list has an id of its own. Where two share
    /// one, a walk cannot say which of them a member is, and hands out
    /// nothing ([`Self::walk`]).
    #[must_use]
    pub fn is_distinct(&self) -> bool {
        self.stands_for.is_empty()
    }

    /// The first entry at or past the position `at`, by its place among the
    /// entries: the place past the last where none is.
    fn start(&self, at: u64) -> usize {
        let slice = (at >> self.shift) as usize;
        let within = self.starts.get(slice..slice + 2).and_then(|ends| {
            let (from, to) = (ends[0] as usize, ends[1] as usize);
            Some((from, self.entries.get(from..to)?))
        });
        match within {
            Some((from, slice)) => from + slice.partition_point(|&(position, _)| position < at),
            // No build leaves a slice out: the search of every entry
            // answers the same.
            None => self.entries.partition_point(|&(position, _)| position < at),
        }
    }

    /// The rows at the ring's positions clockwise from the key's, once
    /// around: the one source of a walk's order. A row stands at every one
    /// of its positions, so a walk meets it more than once.
    fn around(&self, key: u64) -> impl Iterator<Item = u32> + '_ {
        let start = self.start(key_position(key));
        let (before, from) = self.entries.split_at(start);
        from.iter().chain(before).map(|&(_, row)| row)
    }

    /// The rows a request keyed by `key` goes to, by a walk that reads no
    /// row it does not meet: the rows met clockwise from the key are asked
    /// of `keep`, each once, and the first [`MAX_TARGETS`] kept are written
    /// to `out` in the ring's order, the owner first. Returns how many were
    /// written.
    ///
    /// These are the rows the scorer's pass over the whole list ranks, in
    /// its order, where `keep` answers for a row as the profile's filters
    /// do: only the rows read differ.
    ///
    /// `0` where the walk decides nothing and the pass over the whole list
    /// must: once [`DRAW_BOUND`] rows were refused, which says nothing of
    /// the rows behind them, where every row was asked and none kept, and
    /// for a list where two rows share an id ([`Self::is_distinct`]).
    pub fn walk(
        &self,
        key: u64,
        mut keep: impl FnMut(u32) -> bool,
        out: &mut [u32; MAX_TARGETS],
    ) -> usize {
        if !self.is_distinct() {
            return 0;
        }
        // The rows met, kept or refused: a walk meets a row at each of its
        // positions and asks about it once.
        let mut met = [FREE; MAX_TARGETS + DRAW_BOUND as usize];
        let (mut asked, mut kept, mut refused) = (0usize, 0usize, 0u32);
        for row in self.around(key) {
            if met[..asked].contains(&row) {
                continue;
            }
            let Some(slot) = met.get_mut(asked) else {
                continue;
            };
            *slot = row;
            asked += 1;
            if keep(row) {
                out[kept] = row;
                kept += 1;
                if kept == MAX_TARGETS {
                    return kept;
                }
            } else {
                refused += 1;
                if refused == DRAW_BOUND {
                    return 0;
                }
            }
            // Every row was asked: no position further on holds another.
            if asked == self.len() {
                return kept;
            }
        }
        0
    }
}

/// A ring prints its sizes: its entries are a list's rows many times over.
impl fmt::Debug for PoolRing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PoolRing")
            .field("rows", &self.rows)
            .field("entries", &self.entries.len())
            .field("distinct", &self.is_distinct())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::{Ring, WalkStep},
        *,
    };
    use crate::Draw;

    /// A list of `count` workers in an order `shape` draws: each with an id
    /// of its own, spread over the id space, and the id's low half as its
    /// stable key.
    fn list(count: u32, shape: u64) -> Vec<(WorkerId, u32)> {
        let mut draw = Draw::new(shape);
        let mut rows: Vec<(WorkerId, u32)> = (0..count)
            .map(|at| {
                let id = draw.next_u64() | 1;
                (WorkerId(id ^ (u64::from(at) << 40)), id as u32)
            })
            .collect();
        // The list's order is its host's, and no order of the ids.
        for at in (1..rows.len()).rev() {
            rows.swap(at, draw.below(at + 1));
        }
        rows
    }

    /// The scorer's own ring of the same members.
    fn own_ring(rows: &[(WorkerId, u32)]) -> Ring {
        let mut ring = Ring::new();
        ring.add_many(rows.iter().copied());
        ring
    }

    /// The first `MAX_TARGETS` members the scorer's own ring meets from
    /// `key`, by id, each once.
    fn by_own_ring(ring: &Ring, key: u64) -> Vec<WorkerId> {
        let mut met: Vec<WorkerId> = Vec::new();
        ring.walk(key, |member| {
            let id = ring.members()[member as usize];
            if !met.contains(&id) {
                met.push(id);
            }
            if met.len() == MAX_TARGETS.min(ring.len()) {
                WalkStep::Take(())
            } else {
                WalkStep::Skip
            }
        });
        met
    }

    /// The rows a walk of the list's ring hands out from `key` where every
    /// row is kept, by id.
    fn by_list_ring(ring: &PoolRing, rows: &[(WorkerId, u32)], key: u64) -> Vec<WorkerId> {
        let mut out = [0; MAX_TARGETS];
        let handed = ring.walk(key, |_| true, &mut out);
        out[..handed]
            .iter()
            .map(|&row| rows[row as usize].0)
            .collect()
    }

    /// One key has one owner by the list's ring and by the scorer's own
    /// ring of the same members, and the members behind the owner follow in
    /// one order: over lists of every size a gateway holds, in an order of
    /// the host's own, for four thousand keys a list.
    #[test]
    fn a_lists_ring_hands_a_key_the_members_the_scorers_own_ring_hands_it() {
        for (count, shape) in [(1, 1), (2, 2), (3, 3), (8, 4), (64, 5), (512, 6)] {
            let rows = list(count, shape);
            let (ring, own) = (PoolRing::new(rows.iter().copied()), own_ring(&rows));
            assert_eq!((ring.len(), ring.is_distinct()), (count as usize, true));
            let mut keys = Draw::new(shape ^ 0xface);
            for _ in 0..4_000 {
                let key = keys.next_u64();
                assert_eq!(
                    by_list_ring(&ring, &rows, key),
                    by_own_ring(&own, key),
                    "key {key:#x} over {count} members"
                );
            }
        }
    }

    /// Two members whose stable keys are equal stand at the same positions,
    /// every one of them, and the lower id leads at each, by either ring.
    /// The members' 64-bit ids differ and their keys, the ids' low halves,
    /// do not: what two addresses whose fingerprints share a low half give.
    #[test]
    fn members_at_one_position_stand_in_the_order_of_their_ids_by_either_ring() {
        let rows = list(8, 7);
        let matched = rows[2];
        let twin = (WorkerId(matched.0 .0 ^ (1 << 63)), matched.1);
        let (low, high) = (matched.0.min(twin.0), matched.0.max(twin.0));
        // The twin stands ahead of its match in the list, or behind it:
        // the list's order tells nothing.
        for at in [0, rows.len()] {
            let mut rows = rows.clone();
            rows.insert(at, twin);
            let (ring, own) = (PoolRing::new(rows.iter().copied()), own_ring(&rows));
            assert!(ring.is_distinct(), "two ids, one key");
            let mut keys = Draw::new(8);
            let mut led = 0;
            for _ in 0..4_000 {
                let key = keys.next_u64();
                let handed = by_list_ring(&ring, &rows, key);
                assert_eq!(handed, by_own_ring(&own, key), "key {key:#x}");
                if let Some(behind) = handed.iter().position(|id| *id == high) {
                    assert!(
                        behind > 0 && handed[behind - 1] == low,
                        "key {key:#x}: the lower id leads its twin"
                    );
                }
                led += u32::from(handed[0] == low);
            }
            // Nine members at eight sets of positions: the pair owns an
            // eighth of the keys, and the lower id is handed each of them.
            assert!((300..=700).contains(&led), "{led} of 4000 keys");
        }
    }

    /// A walk asks about each row once, hands out the first rows kept in
    /// the ring's order, and says nothing once it met its bound of refused
    /// rows or asked every row and kept none.
    #[test]
    fn a_walk_asks_each_row_once_and_stops_at_its_bounds() {
        let rows = list(64, 9);
        let ring = PoolRing::new(rows.iter().copied());
        let mut keys = Draw::new(10);
        for _ in 0..500 {
            let key = keys.next_u64();
            let whole = by_list_ring(&ring, &rows, key);
            // One row in three refused: the rows handed are the first kept
            // of the ring's order, and no row is asked about twice.
            let refused = |row: u32| row.is_multiple_of(3);
            let mut asked: Vec<u32> = Vec::new();
            let mut out = [0; MAX_TARGETS];
            let handed = ring.walk(
                key,
                |row| {
                    assert!(!asked.contains(&row), "row {row} asked twice");
                    asked.push(row);
                    !refused(row)
                },
                &mut out,
            );
            assert!(asked.len() < MAX_TARGETS + DRAW_BOUND as usize);
            if handed != 0 {
                assert_eq!(handed, MAX_TARGETS);
                let kept: Vec<u32> = asked.iter().copied().filter(|&row| !refused(row)).collect();
                assert_eq!(out[..handed], kept[..]);
                // The owner is the whole walk's first row kept.
                let first = whole.iter().find(|id| {
                    let row = rows.iter().position(|row| row.0 == **id).expect("a row");
                    !refused(row as u32)
                });
                if let Some(first) = first {
                    assert_eq!(rows[out[0] as usize].0, *first);
                }
            }
            // Every row refused: the walk stops at its bound.
            let mut count = 0;
            let none = ring.walk(
                key,
                |_| {
                    count += 1;
                    false
                },
                &mut out,
            );
            assert_eq!((none, count), (0, DRAW_BOUND));
        }
        // A small list is asked about whole, and its rows kept are handed.
        let few = list(3, 11);
        let ring = PoolRing::new(few.iter().copied());
        let mut out = [0; MAX_TARGETS];
        let mut count = 0;
        let handed = ring.walk(
            77,
            |row| {
                count += 1;
                row != 1
            },
            &mut out,
        );
        assert_eq!((handed, count), (2, 3));
        assert!(!out[..handed].contains(&1));
        assert_eq!(ring.walk(77, |_| false, &mut out), 0);
        // No row, no walk.
        let empty = PoolRing::new([]);
        assert!(empty.is_empty());
        assert_eq!(empty.walk(77, |_| true, &mut out), 0);
    }

    /// A list where two rows carry one id is one member twice over: its
    /// ring says so, and a walk of it hands out nothing.
    #[test]
    fn a_list_with_one_id_twice_is_not_walked() {
        let mut rows = list(8, 12);
        rows.push(rows[3]);
        let ring = PoolRing::new(rows.iter().copied());
        assert_eq!((ring.len(), ring.is_distinct()), (9, false));
        let mut out = [0; MAX_TARGETS];
        assert_eq!(ring.walk(5, |_| true, &mut out), 0);
    }

    /// A walk's first entry is found in one slice of the key space, and it
    /// is the entry a search of every entry finds: for positions anywhere,
    /// at an entry, one under it and one over, under the first entry and
    /// past the last.
    #[test]
    fn a_walk_starts_at_the_entry_a_search_of_every_entry_finds() {
        for (count, shape) in [(0, 1), (1, 2), (2, 3), (8, 4), (64, 5), (512, 6)] {
            let ring = PoolRing::new(list(count, shape));
            let search = |at: u64| ring.entries.partition_point(|&(position, _)| position < at);
            let mut draw = Draw::new(shape);
            let mut probes: Vec<u64> = (0..2_000).map(|_| draw.next_u64()).collect();
            probes.extend([0, 1, u64::MAX - 1, u64::MAX]);
            for &(position, _) in ring.entries.iter().step_by(7) {
                probes.extend([position.wrapping_sub(1), position, position.wrapping_add(1)]);
            }
            for at in probes {
                assert_eq!(ring.start(at), search(at), "{at:#x} over {count} rows");
            }
            // Each slice's entries are the ones whose top bits name it.
            let slices = ring.starts.len() - 1;
            assert!(slices.is_power_of_two() && slices >= 2);
            assert_eq!(ring.starts[0], 0);
            assert_eq!(ring.starts[slices] as usize, ring.entries.len());
            assert!(ring.starts.is_sorted());
        }
    }
}
