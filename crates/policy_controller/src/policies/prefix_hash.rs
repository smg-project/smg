//! The `prefix_hash` policy: how it is composed, and how a request's prefix is
//! keyed.
//!
//! The ring and its scorer (`ConsistentHash`) live in `consistent_hashing`:
//! one scorer serves both policies, and it reads the prefix through this file.

use crate::{
    document::spec::{
        BoundedLoadParams, ConsistentHashParams, KeySourceSpec, ProfileSpec, ScoreStage,
        MANUAL_SESSION_HEADER,
    },
    engine::view::{fold_block, RequestFacts},
    policies::{cache_aware::Gates, standard_filters},
};

/// How the `prefix_hash` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    // The ring owner of the request's key: its routing key, else its
    // prefix at 256 tokens. An owner past the bounded-load rule is
    // passed over and a request with no key has no owner: the
    // fewest requests in flight take both. The ring's weight is by
    // arithmetic: an owner leads the ring's next member by a
    // quarter of the stage's score and the count ranges over one
    // whole score, so any weight past four keeps a kept owner above
    // the least loaded worker. The pick names no sample: the owner
    // must be among the rows scored, so the pool is read whole.
    ProfileSpec {
        filter: standard_filters(),
        score: vec![
            ScoreStage::ConsistentHash(ConsistentHashParams {
                weight: 8.0,
                prefix_tokens: Some(256),
                bounded_load: Some(BoundedLoadParams {
                    factor: Some(1.25),
                    abs_margin: Some(10),
                }),
            })
            .into(),
            "least-requests".into(),
        ],
        pick: "best".into(),
        sent_ttl_ms: None,
    }
}

/// The headers the `prefix_hash` policy reads a request's routing key from, in
/// order. `header` makes one source from a header's name and whether the
/// header tells one client from another.
pub(crate) fn key_sources(header: impl Fn(&str, bool) -> KeySourceSpec) -> Vec<KeySourceSpec> {
    vec![header(MANUAL_SESSION_HEADER, false)]
}

/// The key of a request's prefix at a depth of `tokens`: the fold of its
/// block hashes through its last whole block at or under that depth, `0`
/// where it carries no block that shallow. A request shorter than the
/// depth is keyed through its own last whole block, and so is one whose
/// hashes the host capped short of the depth.
pub(crate) fn prefix_key(tokens: u32, gates: Gates, facts: &RequestFacts<'_>) -> u64 {
    let depth = gates.whole_blocks(tokens, facts) as usize;
    match depth.min(facts.block_hashes.len()) {
        0 => 0,
        blocks => folded(facts, blocks),
    }
}

/// The key of the request's first `blocks` blocks, one at least, which it
/// carries: the fold of their hashes under its cache namespace
/// ([`fold_block`]), one multiplication a block.
// Out of line: a request with nothing to hash carries no part of the loop.
#[inline(never)]
fn folded(facts: &RequestFacts<'_>, blocks: usize) -> u64 {
    let hashes = facts.block_hashes[..blocks].iter();
    hashes.fold(facts.namespace_seed, |key, &hash| fold_block(key, hash))
}

/// The prefix form of the ring: how a request with no routing key is keyed.
#[derive(Debug)]
pub(crate) struct Prefix {
    /// The depth in tokens such a request is keyed at when it reaches no
    /// boundary.
    pub(crate) tokens: u32,
    /// The gates that hold the view's block size.
    pub(crate) gates: Gates,
    /// The view's boundaries in tokens, ascending.
    pub(crate) boundaries: Box<[u32]>,
}

impl Prefix {
    /// The request's prefix at `boundary`, where it reaches it: the fold
    /// through the boundary's last whole block ([`folded`]), made when the
    /// boundary's turn comes. A request reaches a boundary by carrying
    /// that block.
    pub(crate) fn at(&self, boundary: u32, facts: &RequestFacts<'_>) -> Option<u64> {
        let blocks = self.gates.whole_blocks(boundary, facts) as usize;
        let reached = blocks != 0 && blocks <= facts.block_hashes.len();
        reached.then(|| folded(facts, blocks))
    }
}
