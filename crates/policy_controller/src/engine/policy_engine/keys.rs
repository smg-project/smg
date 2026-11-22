//! A keyed request's decision over a gateway's own list, by the rows its
//! key names.
//!
//! The engine decides a keyed request of the single plan over rows read one
//! at a time ([`Engine::decide_keyed`]): its key's pin is one row, and a
//! ring's walk is the key's owner and the members behind it. Here those rows
//! are the records of a gateway's own list, each written when the decision
//! asks about its worker ([`PoolRecords::read`]), and the ring is the
//! gateway's own: the one its classic policies walk, which the request's
//! maker hands ([`RequestMaker::ring`]) and the list reads through a table
//! of its own ([`ring_of`]).
//!
//! No decision builds a ring, and none waits for one: the gateway fills a
//! membership's ring off the path of a request. A request that reads a ring
//! where the gateway holds none yet is placed by the gateway's rule for a
//! request with no ring ([`RequestFacts::ring_awaited`]).

use std::{cell::OnceCell, sync::Arc};

use smg_worker::identity::basic::Worker;

use super::PoolMembers;
use crate::{
    engine::{
        decision::Decision,
        decision_engine::{MembershipRing, RequestMaker},
        framework::{Engine, PoolRows},
        records::{Live, PoolRecords},
        view::{Candidate, Constraints, RequestFacts, WorkerId},
    },
    policies::consistent_hashing::{ListRing, RingPlaces},
};

/// Where a request's decision finds its host's hash ring: the request's
/// maker, which hands it, and the place it is read into as the request's
/// list reads it.
///
/// The maker is asked at most once a request, and only where a stage of
/// the request's binding reads a ring ([`Engine::decide_keyed`]): a request
/// of any other binding asks its host for none, and nothing is made for it.
#[derive(Clone, Copy)]
pub(super) struct LazyRing<'r> {
    /// The request's maker, asked for the ring its host holds for the
    /// request's list ([`RequestMaker::ring`]).
    pub(super) maker: &'r dyn RequestMaker,
    /// The ring as the list reads it ([`ring_of`]), once it was read:
    /// `None` inside where the host holds none. It stands ahead of the
    /// request, whose facts borrow it.
    pub(super) cell: &'r OnceCell<Option<ListRing<'r>>>,
}

/// One list of a gateway's worker objects as a keyed decision reads it: the
/// list's records, one worker's written at a time, and the gateway's ring as
/// the list reads it.
struct ListRows<'r, 'a> {
    ring: LazyRing<'r>,
    members: &'r PoolMembers,
    records: &'r PoolRecords,
    workers: &'a [Arc<dyn Worker>],
    live: &'a Live<'a>,
}

impl<'r> PoolRows<'r> for ListRows<'r, '_> {
    fn len(&self) -> usize {
        self.records.len()
    }

    fn read(&self, place: u32) -> Option<Candidate> {
        self.records.read(self.workers, self.live, place)
    }

    fn place_of(&self, id: WorkerId) -> Option<u32> {
        self.records.place_of(id)
    }

    fn generation(&self) -> u64 {
        self.records.generation()
    }

    // The ring the request's host holds for the list, asked for and read as
    // the list reads it at the first call.
    fn ring(&self) -> Option<&'r ListRing<'r>> {
        let LazyRing { maker, cell } = self.ring;
        cell.get_or_init(|| ring_of(maker.ring()?, self.members, self.records))
            .as_ref()
    }

    fn keeps_a_ring(&self) -> bool {
        true
    }
}

/// `handed`, the ring a request's host handed with it, as `members` reads
/// it: the ring, and the place in the list of the row at each of its
/// addresses.
///
/// The ring of the list's own membership is read through the table kept
/// with the list ([`PoolMembers::ring_places`]). The list's first decision
/// behind that ring's fill makes the table, once: it searches the list's
/// ids for each address of the ring and allocates the table. Every later
/// decision reads it, and makes nothing.
///
/// A ring of another membership, which a host hands between a membership's
/// change and the fill of the ring that follows it, is read with no table:
/// each entry of the ring a walk meets is sought among the list's ids
/// ([`ListRing::by_search`]). Nothing is made for it, and nothing kept: the
/// list keeps the table of its own ring alone.
///
/// `None` for such a ring over a list where two rows carry one id: no order
/// of its ids names its rows ([`PoolRecords::by_id`]), and its request is
/// placed as one whose host holds no ring yet. A registry holds one worker
/// an address, so no list it publishes is such a list.
fn ring_of<'r>(
    handed: MembershipRing<'r>,
    members: &'r PoolMembers,
    records: &'r PoolRecords,
) -> Option<ListRing<'r>> {
    if handed.of_this_membership {
        let places = members
            .ring_places()
            .get_or_init(|| RingPlaces::new(handed.ring, records.ids()));
        // A list has one membership and so one ring of its own: a second
        // ring handed as its own is its host's error, read as a ring of
        // another membership.
        if places.is_of(handed.ring) {
            return Some(places.list_ring());
        }
    }
    records
        .has_distinct_ids()
        .then(|| ListRing::by_search(handed.ring, records.len(), records.by_id()))
}

/// A keyed request's decision over `members`, the list `records` were made
/// from, by the rows its key names: its key's pin under the binding's
/// sessions rule, or its key's owner on `ring`, its host's ring as the list
/// reads it, and the members behind it. `None` where the key decides
/// nothing by those rows: the caller then asks the whole decision
/// ([`records::decide`]) with the same `facts`, which carry the ring from
/// here where the binding reads one, or the word that its host holds none
/// yet.
///
/// The decision's instant is written into `facts` here
/// ([`RequestFacts::now_micros`]), the one `live` was made at.
///
/// [`records::decide`]: crate::engine::records::decide
pub(super) fn decide<'r>(
    engine: &Engine,
    ring: LazyRing<'r>,
    members: &'r PoolMembers,
    records: &'r PoolRecords,
    live: &Live<'_>,
    facts: &mut RequestFacts<'r>,
    cx: &Constraints<'_>,
) -> Option<Decision> {
    facts.now_micros = live.now_micros();
    let rows = ListRows {
        ring,
        members,
        records,
        workers: members.workers(),
        live,
    };
    engine.decide_keyed(facts, &rows, cx)
}

#[cfg(test)]
mod tests;
