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
    policies::{cache_aware::Gates, consistent_hashing::Head, standard_filters},
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
        sent_report_delay_ms: None,
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

/// The characters of a text that count as one token where a head of it is
/// cut, as the gateway's own policy counts them
/// (`model_gateway/src/policies/prefix_hash.rs`, line 119 at 882be6f4f).
const CHARS_PER_TOKEN: usize = 4;

/// How a prefix-form stage cuts a request into the heads it is placed by
/// on a ring its caller lends
/// ([`Scorer::places_by_heads`](crate::engine::framework::Scorer::places_by_heads)).
///
/// The rule gains fields: code outside this crate makes it by
/// [`HeadRule::new`], and matches it with `..`.
///
/// ```
/// use smg_policy_controller::policies::prefix_hash::HeadRule;
/// let rule = HeadRule::new(256, &[64, 128]);
/// assert_eq!((rule.depth, rule.boundaries), (256, &[64, 128][..]));
/// ```
///
/// ```compile_fail
/// use smg_policy_controller::policies::prefix_hash::HeadRule;
/// let _ = HeadRule { depth: 256, boundaries: &[64, 128] };
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct HeadRule<'a> {
    /// The depth in tokens of the one head of a prompt that reaches no
    /// boundary.
    pub depth: u32,
    /// The view's boundaries in tokens, ascending: a prompt has a head at
    /// each one it reaches.
    pub boundaries: &'a [u32],
}

impl<'a> HeadRule<'a> {
    /// The rule of a head `depth` tokens deep and of the view's
    /// `boundaries`, ascending.
    #[must_use]
    pub const fn new(depth: u32, boundaries: &'a [u32]) -> Self {
        Self { depth, boundaries }
    }
}

impl HeadRule<'_> {
    /// The heads of a prompt as the gateway's own `prefix_hash` asks its
    /// ring about them, each handed to `state`, the shallowest first
    /// (`model_gateway/src/policies/prefix_hash.rs`, lines 143 to 205 and
    /// 361 to 386 at 882be6f4f). Token ids lead a text, and an empty list
    /// of either is none. A prompt has a head at each boundary it reaches.
    /// One that reaches none has one head, of the rule's depth, and is its
    /// own head where it is shorter. A text is measured at four characters
    /// a token and cut on a character's edge. A prompt with nothing to
    /// read has no head. The gateway tries the deepest head
    /// first: what states the heads' places turns the list round, as the
    /// one call does, and a host that writes a request's facts itself
    /// cuts its heads here.
    pub fn heads<'a>(
        self,
        tokens: Option<&'a [u32]>,
        text: Option<&'a str>,
        mut state: impl FnMut(Head<'a>),
    ) {
        let mut stated = false;
        match (tokens, text) {
            (Some(ids), _) if !ids.is_empty() => {
                let reached = |&&at: &&u32| at as usize <= ids.len();
                for &at in self.boundaries.iter().filter(reached) {
                    stated = true;
                    state(Head::Tokens(&ids[..at as usize]));
                }
                if !stated {
                    let depth = ids.len().min(self.depth as usize);
                    state(Head::Tokens(&ids[..depth]));
                }
            }
            (_, Some(text)) if !text.is_empty() => {
                let mut chars = text.char_indices();
                let mut read = 0usize;
                for &at in self.boundaries {
                    let budget = (at as usize).saturating_mul(CHARS_PER_TOKEN);
                    // The boundary's last character, where the text holds it.
                    if budget > read && chars.nth(budget - read - 1).is_none() {
                        break;
                    }
                    // A boundary that does not ascend leaves the text
                    // where it was read to, as the gateway's own cut does.
                    read = read.max(budget);
                    stated = true;
                    let behind = chars.clone().next();
                    let end = behind.map_or(text.len(), |(offset, _)| offset);
                    state(Head::Text(&text[..end]));
                }
                if !stated {
                    let budget = (self.depth as usize).saturating_mul(CHARS_PER_TOKEN);
                    let behind = text.char_indices().nth(budget);
                    let end = behind.map_or(text.len(), |(offset, _)| offset);
                    state(Head::Text(&text[..end]));
                }
            }
            _ => {}
        }
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The heads `rule` states of a prompt, in the order it states them.
    fn stated<'a>(
        rule: HeadRule<'_>,
        tokens: Option<&'a [u32]>,
        text: Option<&'a str>,
    ) -> Vec<Head<'a>> {
        let mut heads = Vec::new();
        rule.heads(tokens, text, |head| heads.push(head));
        heads
    }

    /// A prompt is cut where the gateway's own policy cuts it
    /// (`model_gateway/src/policies/prefix_hash.rs`, lines 143 to 205 at
    /// 882be6f4f): a head at each boundary it reaches, the shallowest
    /// first, where the last token of a boundary reaches it. A prompt that
    /// reaches none has the one head of the rule's depth, and is its own
    /// head where it is shorter. A text is measured at four characters a
    /// token and cut on a character's edge, whatever bytes a character
    /// takes. Token ids lead a text, an empty list of ids leaves the text,
    /// and a prompt with nothing to read has no head.
    #[test]
    fn a_prompt_has_the_heads_the_gateways_own_policy_cuts() {
        let rule = HeadRule {
            depth: 128,
            boundaries: &[64, 256],
        };
        let ids: Vec<u32> = (0..300).collect();
        let tokens = |len: usize| Head::Tokens(&ids[..len]);
        assert_eq!(stated(rule, Some(&ids), None), [tokens(64), tokens(256)]);
        let at_the_last = stated(rule, Some(&ids[..256]), None);
        assert_eq!(at_the_last, [tokens(64), tokens(256)]);
        assert_eq!(stated(rule, Some(&ids[..255]), None), [tokens(64)]);
        assert_eq!(stated(rule, Some(&ids[..63]), None), [tokens(63)]);
        let deep = HeadRule {
            depth: 128,
            boundaries: &[256],
        };
        assert_eq!(stated(deep, Some(&ids[..200]), None), [tokens(128)]);

        // A text: boundaries at eight and sixteen characters, a depth of
        // twelve.
        let rule = HeadRule {
            depth: 3,
            boundaries: &[2, 4],
        };
        let text = "abcdefghijklmnopq";
        let cut = |len: usize| Head::Text(&text[..len]);
        assert_eq!(stated(rule, None, Some(text)), [cut(8), cut(16)]);
        assert_eq!(stated(rule, None, Some(&text[..16])), [cut(8), cut(16)]);
        assert_eq!(stated(rule, None, Some(&text[..15])), [cut(8)]);
        assert_eq!(stated(rule, None, Some(&text[..7])), [cut(7)]);
        let unbounded = HeadRule {
            depth: 3,
            boundaries: &[],
        };
        assert_eq!(stated(unbounded, None, Some(text)), [cut(12)]);
        // The rule's depth counts characters too: twelve of two bytes each.
        let wide = "\u{e9}".repeat(14);
        let twelve = [Head::Text(&wide[..24])];
        assert_eq!(stated(unbounded, None, Some(&wide)), twelve);
        // Boundaries a validated document never holds, out of order: the
        // gateway's own cut reads on from where the text was read to
        // (the same file, lines 185 to 205).
        let unordered = HeadRule {
            depth: 1,
            boundaries: &[2, 1, 3],
        };
        let cuts = [cut(8), cut(8), cut(12)];
        assert_eq!(stated(unordered, None, Some(&text[..16])), cuts);
        // Nine characters of two bytes each: the head of eight holds
        // sixteen bytes.
        let wide = "\u{e9}".repeat(9);
        assert_eq!(stated(rule, None, Some(&wide)), [Head::Text(&wide[..16])]);

        assert_eq!(
            stated(rule, Some(&ids[..5]), Some(text)),
            [tokens(2), tokens(4)]
        );
        assert_eq!(stated(rule, Some(&[]), Some(&text[..7])), [cut(7)]);
        assert!(stated(rule, Some(&[]), Some("")).is_empty());
        assert!(stated(rule, None, None).is_empty());
    }
}
