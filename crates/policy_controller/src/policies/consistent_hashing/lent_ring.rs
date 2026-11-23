//! The hash ring a request's caller keeps, as a decision asks it
//! ([`RequestFacts::ring`]). The consistent-hash scorer asks the ring which
//! workers the request's key reaches, in the ring's order: a key has the
//! owner its caller's own walk finds. Nothing of such a ring is made here:
//! no position is hashed and no point is kept.
//!
//! [`RequestFacts::ring`]: crate::engine::RequestFacts::ring

use std::fmt;

/// A hash ring of workers that a request's caller keeps and lends to a
/// decision. It names a worker by the address it is registered at, and a
/// decision finds the worker's row by the id a gateway's list states for
/// that address ([`WorkerId::from_address`]): one ring serves every list
/// of its workers, and an address no row of a list is at is passed over.
///
/// [`WorkerId::from_address`]: crate::engine::WorkerId::from_address
pub trait LentRing: fmt::Debug + Send + Sync {
    /// Where `key` stands on the ring: the ring's own hash of the key's
    /// text. A caller states it for the request's key. A ring that holds
    /// no worker answers too: the plain ring places its key by the
    /// position alone, and the prefix form leaves it to the stages behind
    /// the ring.
    fn position(&self, key: &str) -> u64;

    /// The walk from `position`: `take` is asked about the ring's workers,
    /// each by its address and each once, in the ring's order clockwise
    /// from the position, until it takes one or every worker was asked.
    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool);
}

/// A ring for the crate's cases, of the addresses it holds in the order
/// stated: a walk starts as many workers along that order as its position
/// says, and goes round once.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct InOrder(pub(crate) Vec<String>);

#[cfg(test)]
impl LentRing for InOrder {
    fn position(&self, key: &str) -> u64 {
        crate::engine::fingerprint(key.as_bytes())
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let (before, behind) = self.0.split_at(position as usize % self.0.len().max(1));
        let _ = behind.iter().chain(before).any(|address| take(address));
    }
}
