//! The hash ring a request's caller keeps, as a decision asks it
//! ([`RequestFacts::ring`]). The consistent-hash scorer asks the ring which
//! workers the request's key reaches, in the ring's order: a key has the
//! owner its caller's own walk finds. Nothing of such a ring is made here:
//! no position is hashed and no point is kept.
//!
//! [`RequestFacts::ring`]: crate::engine::RequestFacts::ring

use std::fmt;

/// What a lender's ring is asked the place of under the prefix form of the
/// ring ([`LentRing::head_position`]): a head of the request's prompt, or
/// its routing value. A head is a part of the request's own ids or text:
/// nothing is copied for it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Head<'a> {
    /// The first token ids of a tokenized prompt. The bytes a lender's
    /// recipe reads are each id's four bytes in the machine's order, one id
    /// after the other: the gateway's own policy casts the ids in place
    /// (`model_gateway/src/policies/prefix_hash.rs`, lines 145 and 178 at
    /// 882be6f4f).
    Tokens(&'a [u32]),
    /// The leading text of a prompt that carries no token ids, cut on a
    /// character's edge, and the whole text of a routing value, which
    /// leads every prompt (the same file, lines 358 to 360). A recipe
    /// reads its bytes.
    Text(&'a str),
}

/// Sizes alone: a client's tokens and text reach no log line.
impl fmt::Debug for Head<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Tokens(ids) => write!(f, "Tokens({})", ids.len()),
            Self::Text(text) => write!(f, "Text({})", text.len()),
        }
    }
}

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

    /// Where `head` stands on the ring, by the lender's own recipe for a
    /// head, and `None` from a lender whose program places no request by
    /// one. The gateway's own `prefix_hash` takes a hash of the head's
    /// bytes, spells it as sixteen lowercase hex digits and asks its ring
    /// where that text stands (`model_gateway/src/policies/prefix_hash.rs`,
    /// lines 223 to 229 and 291 to 293 at 882be6f4f): a lender that answers
    /// by its program's recipe gives a prompt the owner its own walk finds.
    /// No part of a recipe is held here, and nothing is hashed for one.
    ///
    /// What makes a request under the prefix form asks once for each head
    /// of the prompt, or once for the routing value, and states the
    /// answers as the request's places, the deepest head's first
    /// ([`RequestFacts::ring_positions`]). The order of the asks is its
    /// own. Under `None` the request is lent no ring and states no place,
    /// so the stage's own ring places it. The lender is then asked about
    /// no further head, and [`Self::position`] is asked nothing for a
    /// head: one recipe's place is never read under the other form. A ring
    /// that holds no worker answers too, as [`Self::position`] does.
    ///
    /// [`RequestFacts::ring_positions`]: crate::engine::RequestFacts::ring_positions
    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        let _ = head;
        None
    }

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

    // Another recipe than a key's text has: the fingerprint of the head's
    // bytes turned half round, so a case that asks the one place for the
    // other reads another number.
    fn head_position(&self, head: Head<'_>) -> Option<u64> {
        let bytes = match head {
            Head::Tokens(ids) => ids.iter().flat_map(|id| id.to_ne_bytes()).collect(),
            Head::Text(text) => text.as_bytes().to_vec(),
        };
        Some(crate::engine::fingerprint(&bytes).rotate_left(32))
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let (before, behind) = self.0.split_at(position as usize % self.0.len().max(1));
        let _ = behind.iter().chain(before).any(|address| take(address));
    }
}

#[cfg(test)]
mod tests {
    use super::Head;

    /// A head's debug form says its size and nothing it holds: a client's
    /// token ids and text reach no log line through it.
    #[test]
    fn a_heads_debug_form_says_its_size_alone() {
        assert_eq!(format!("{:?}", Head::Text("a client's words")), "Text(16)");
        assert_eq!(format!("{:?}", Head::Tokens(&[7, 8, 9])), "Tokens(3)");
    }
}
