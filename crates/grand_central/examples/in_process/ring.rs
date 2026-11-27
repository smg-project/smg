//! The program's own hash ring.
//!
//! A program that routes by a ring of its own keeps one for each membership
//! generation of a model, and lends it with every request ([`LentRing`]).
//!
//! Under a document whose ring stage places a routing value by its text, the
//! value goes to the worker this ring's walk reaches from where the value
//! stands. So it stays where the program's own ring policy puts it.
//!
//! The program's ring places no request by a head of its prompt, so it answers
//! no place for one ([`LentRing::head_position`], as provided). The prefix form
//! of the ring then places such a request by its own ring. Under any other
//! document the engine asks the ring for an encode request's items alone.
//!
//! The program makes its ring with each generation and lends it always. A host
//! whose ring is costly makes it only where its document places by one. The
//! document's needs say so (`Needs::ring`), and a gateway asks them before it
//! reads its own ring.
//!
//! The recipe is the program's own, and the engine reads none of it. On this
//! ring a worker stands once, where a hash of its address puts it. A gateway's
//! ring gives each worker many places.

use smg_grand_central::{engine::WorkerId, policies::consistent_hashing::LentRing};

/// One worker of the ring: where it stands, its address, and the id a list
/// states for that address.
#[derive(Debug)]
struct Node {
    at: u64,
    address: Box<str>,
    id: WorkerId,
}

/// The ring of one generation: its workers in the order of their places.
#[derive(Debug)]
pub struct Ring(Box<[Node]>);

/// Where `text` stands on the ring: a 64-bit FNV-1a hash of its bytes, mixed
/// once more so that texts that differ in their last letters stand far
/// apart. It is the same in every process.
fn place(text: &str) -> u64 {
    const OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01b3;
    let hash = text.bytes().fold(OFFSET, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(PRIME)
    });
    let hash = (hash ^ (hash >> 33)).wrapping_mul(0xff51_afd7_ed55_8ccd);
    let hash = (hash ^ (hash >> 33)).wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    hash ^ (hash >> 33)
}

impl Ring {
    /// The ring of the workers registered at `addresses`.
    pub fn of<'a>(addresses: impl Iterator<Item = &'a str>) -> Self {
        let node = |address: &str| Node {
            at: place(address),
            address: address.into(),
            id: WorkerId::from_address(address),
        };
        let mut nodes: Vec<Node> = addresses.map(node).collect();
        nodes.sort_unstable_by(|one, other| {
            let (one, other) = ((one.at, &one.address), (other.at, &other.address));
            one.cmp(&other)
        });
        Self(nodes.into())
    }

    /// The workers in the order a walk from `position` asks about them:
    /// from the first that stands at the position or past it, round once.
    fn from(&self, position: u64) -> impl Iterator<Item = &Node> {
        let start = self.0.partition_point(|node| node.at < position);
        let (before, behind) = self.0.split_at(start);
        behind.iter().chain(before)
    }

    /// The addresses a walk from where `value` stands asks about, in the
    /// walk's order: the first is the worker the ring puts the value on.
    pub fn walked(&self, value: &str) -> impl Iterator<Item = &str> {
        self.from(place(value)).map(|node| &*node.address)
    }
}

impl LentRing for Ring {
    fn position(&self, key: &str) -> u64 {
        place(key)
    }

    fn walk(&self, position: u64, take: &mut dyn FnMut(&str) -> bool) {
        let _ = self.from(position).any(|node| take(&node.address));
    }

    // The ids were hashed where the ring was made: a decision hashes none.
    fn walk_ids(&self, position: u64, take: &mut dyn FnMut(WorkerId) -> bool) {
        let _ = self.from(position).any(|node| take(node.id));
    }
}
