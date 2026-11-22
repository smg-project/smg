//! A membership's hash ring as one list of its workers reads it.
//!
//! A host that keeps its workers in lists of its own, as a gateway that
//! decides in its own process does, keeps one hash ring a membership
//! ([`HashRing`]): the ring its classic policies walk. The ring names a
//! worker by its address, and a decision names one by its row's place in
//! the list it is decided over. What stands between the two is one table:
//! for each address of the ring, the place of that address's row in the
//! list ([`RingPlaces`]).
//!
//! The host lends the ring with that table to each decision of the list
//! ([`RequestFacts::ring`]), and says where the request's key stands on it
//! ([`RequestFacts::ring_position`]). The consistent-hash scorer then walks
//! the ring's own entries from that position, as the host's classic policy
//! does, and its own ring ([`Ring`]) learns nothing of the decision. No
//! second ring is built: a key has the owner the host's ring gives it.
//!
//! The ring a host hands can be of another membership than the list, kept
//! while the list's own is filled. No table is made of such a ring: the row
//! at each address a walk meets is sought among the list's ids
//! ([`ListRing::by_search`]). An address the list holds no row of is passed
//! over, and a row whose address the ring does not hold is met by no walk:
//! what the host's classic policy answers on a kept ring.
//!
//! [`RequestFacts::ring`]: crate::engine::RequestFacts::ring
//! [`RequestFacts::ring_position`]: crate::engine::RequestFacts::ring_position
//! [`Ring`]: super::Ring

use std::{fmt, sync::Arc};

use super::{hash_ring::HashRing, FREE};
use crate::engine::{decision::MAX_TARGETS, framework::DRAW_BOUND, view::WorkerId};

/// Where the addresses of one hash ring stand in one list of workers: for
/// the ring's address at each index, the place of that address's row.
///
/// Made once for a list and the ring of the list's own membership, and kept
/// with the list for as long as it stands. Neither changes, so a reader
/// takes no lock. It holds the ring it was made for: a decision handed
/// another ring is not read through it ([`Self::is_of`]).
///
/// Two rows of one id are one member: the first of them stands for both
/// ([`ListRing::has_distinct_ids`]).
pub struct RingPlaces {
    ring: Arc<HashRing>,
    /// The place of the row at each of the ring's addresses, by the ring's
    /// index of the address; [`FREE`] for an address no row of the list is
    /// at.
    places: Box<[u32]>,
    /// Where two rows of the list carry one id: for each row, the first row
    /// of its id. Empty where every row has an id of its own.
    stands_for: Box<[u32]>,
    /// How many rows the list holds.
    rows: u32,
}

impl RingPlaces {
    /// The places of `ring`'s addresses in a list whose rows carry `ids`, in
    /// the list's order.
    ///
    /// Called once a list, off the path of every later decision. It
    /// allocates, sorts the ids and searches them once for each address of
    /// the ring. Nothing is hashed but each address's id
    /// ([`WorkerId::from_address`]).
    #[must_use]
    pub fn new(ring: &Arc<HashRing>, ids: &[WorkerId]) -> Self {
        // The rows by id, a row that repeats an id behind the first of it.
        let mut by_id: Vec<(WorkerId, u32)> = ids.iter().copied().zip(0u32..).collect();
        by_id.sort_unstable();
        let mut stands_for: Vec<u32> = Vec::new();
        let mut first: Option<(WorkerId, u32)> = None;
        by_id.retain(|&(id, at)| match first {
            Some((standing, member)) if standing == id => {
                if stands_for.is_empty() {
                    stands_for.extend(0..ids.len() as u32);
                }
                stands_for[at as usize] = member;
                false
            }
            _ => {
                first = Some((id, at));
                true
            }
        });
        let places = ring
            .urls()
            .iter()
            .map(|address| row_of(&by_id, WorkerId::from_address(address)).unwrap_or(FREE))
            .collect();
        Self {
            ring: Arc::clone(ring),
            places,
            stands_for: stands_for.into(),
            rows: ids.len() as u32,
        }
    }

    /// Whether these are the places of `ring`'s addresses: the ring they
    /// were made for, by its identity.
    #[must_use]
    pub fn is_of(&self, ring: &HashRing) -> bool {
        std::ptr::eq(&*self.ring, ring)
    }

    /// The ring as the list reads it, for a decision.
    #[must_use]
    pub fn list_ring(&self) -> ListRing<'_> {
        ListRing {
            ring: &self.ring,
            places: Places::Table(&self.places),
            stands_for: &self.stands_for,
            rows: self.rows,
        }
    }
}

/// The places print their sizes: the table is one word an address.
impl fmt::Debug for RingPlaces {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RingPlaces")
            .field("rows", &self.rows)
            .field("addresses", &self.places.len())
            .field("distinct", &self.stands_for.is_empty())
            .finish_non_exhaustive()
    }
}

/// The place of the row whose id is `id`, among `by_id`: a list's rows by
/// id, ids ascending and each once, with each row's place.
fn row_of(by_id: &[(WorkerId, u32)], id: WorkerId) -> Option<u32> {
    let at = by_id.binary_search_by_key(&id, |&(id, _)| id).ok()?;
    by_id.get(at).map(|&(_, place)| place)
}

/// How a list finds the row at one of a ring's addresses.
#[derive(Clone, Copy)]
enum Places<'a> {
    /// By the list's table of the ring ([`RingPlaces`]): the place of the
    /// row at each address, by the ring's index of the address.
    Table(&'a [u32]),
    /// With no table, by the address's id among the list's rows by id, ids
    /// ascending and each once ([`ListRing::by_search`]).
    Search(&'a [(WorkerId, u32)]),
}

/// A hash ring as a decision over one list of workers walks it: the ring,
/// and the place in the list of the row at each of its addresses.
///
/// A member is named by its row's place in the list: a walk hands out
/// places, and its caller reads the row there with no search for it.
///
/// It borrows what it finds a row by: the table kept with the list
/// ([`RingPlaces::list_ring`]), or the list's ids where the ring handed is one
/// the list keeps no table of ([`Self::by_search`]).
#[derive(Clone, Copy)]
pub struct ListRing<'a> {
    ring: &'a HashRing,
    places: Places<'a>,
    /// Where two rows of the list carry one id: for each row, the first row
    /// of its id. Empty where every row has an id of its own.
    stands_for: &'a [u32],
    rows: u32,
}

impl<'a> ListRing<'a> {
    /// `ring` as a list of `rows` rows reads it with no table of it:
    /// `by_id` is the list's rows by id, ids ascending and each once, with
    /// each row's place.
    ///
    /// For a decision handed the ring of another membership, between a
    /// membership's change and the fill of the list's own ring. Nothing is
    /// made for it. Each entry of the ring a walk meets then costs one
    /// hash of its address and one search of `by_id`, where a table costs
    /// one read. A list where two rows carry one id is not read so: no
    /// `by_id` names its rows once each.
    #[must_use]
    pub fn by_search(ring: &'a HashRing, rows: usize, by_id: &'a [(WorkerId, u32)]) -> Self {
        Self {
            ring,
            places: Places::Search(by_id),
            stands_for: &[],
            rows: rows as u32,
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
    /// nothing ([`Self::walk`]): such a list is placed by the scorer's pass
    /// over its rows, which takes the last of them the filters keep.
    #[must_use]
    pub fn has_distinct_ids(&self) -> bool {
        self.stands_for.is_empty()
    }

    /// The place of the row at the ring's address `address`, and `None`
    /// where the list holds no row of it.
    #[inline(always)]
    fn row_at(&self, address: u32) -> Option<u32> {
        match self.places {
            Places::Table(places) => places
                .get(address as usize)
                .copied()
                .filter(|&row| row != FREE),
            Places::Search(by_id) => {
                let address = self.ring.urls().get(address as usize)?;
                row_of(by_id, WorkerId::from_address(address))
            }
        }
    }

    /// The rows at the ring's positions clockwise from `position`, once
    /// around: the one source of a walk's order. A row stands at every one
    /// of its address's positions, so a walk meets it more than once. An
    /// address the list holds no row of is passed over.
    pub(super) fn around(&self, position: u64) -> impl Iterator<Item = u32> + '_ {
        self.ring
            .around(position)
            .filter_map(|address| self.row_at(address))
    }

    /// The rows a request whose key stands at `position` goes to, by a walk
    /// that reads no row it does not meet: the rows met clockwise from the
    /// position are asked of `keep`, each once, and the first
    /// [`MAX_TARGETS`] kept are written to `out` in the ring's order, the
    /// owner first. Returns how many were written.
    ///
    /// These are the rows the scorer's pass over the whole list ranks, in
    /// its order, where `keep` answers for a row as the profile's filters
    /// do: only the rows read differ. A walk that came round the ring hands
    /// out the rows it kept: a row whose address the ring does not hold
    /// takes no key by it.
    ///
    /// `0` where the walk decides nothing and the pass over the whole list
    /// must: once [`DRAW_BOUND`] rows were refused, which says nothing of
    /// the rows behind them, where no row met was kept, and for a list
    /// where two rows share an id ([`Self::has_distinct_ids`]).
    pub fn walk(
        &self,
        position: u64,
        mut keep: impl FnMut(u32) -> bool,
        out: &mut [u32; MAX_TARGETS],
    ) -> usize {
        if !self.has_distinct_ids() {
            return 0;
        }
        // The rows met, kept or refused: a walk meets a row at each of its
        // positions and asks about it once.
        let mut met = [FREE; MAX_TARGETS + DRAW_BOUND as usize];
        let (mut asked, mut kept, mut refused) = (0usize, 0usize, 0u32);
        for row in self.around(position) {
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
        kept
    }

    /// Fills `slots`, a slot a row of the list, with the index among `kept`
    /// of each row kept, and the mark of a free slot for every other.
    /// `kept` names rows by their places, ascending. Two rows of one id
    /// share the slot of the first of them, and the last of them kept takes
    /// it, as in the scorer's own ring.
    pub(super) fn fill(&self, kept: &[u32], slots: &mut Vec<u32>) {
        slots.clear();
        if self.has_distinct_ids() {
            // Every row kept: each row's slot is its own place.
            if kept.len() == self.len() {
                slots.extend(0..self.rows);
                return;
            }
            slots.resize(self.len(), FREE);
            for (at, &row) in (0u32..).zip(kept) {
                if let Some(slot) = slots.get_mut(row as usize) {
                    *slot = at;
                }
            }
        } else {
            slots.resize(self.len(), FREE);
            for (at, &row) in (0u32..).zip(kept) {
                let member = self.stands_for.get(row as usize);
                if let Some(slot) = member.and_then(|&member| slots.get_mut(member as usize)) {
                    *slot = at;
                }
            }
        }
    }
}

/// A list's ring prints its sizes: its entries are a membership's addresses
/// many times over.
impl fmt::Debug for ListRing<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ListRing")
            .field("rows", &self.rows)
            .field("addresses", &self.ring.worker_count())
            .field("entries", &self.ring.len())
            .field("table", &matches!(self.places, Places::Table(_)))
            .field("distinct", &self.has_distinct_ids())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests;
