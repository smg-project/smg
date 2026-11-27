//! The `round_robin` policy: how it is composed, and the picker only it uses.

use std::{
    hash::Hasher,
    sync::atomic::{AtomicU64, Ordering},
};

use rustc_hash::FxHasher;

use crate::{
    document::spec::ProfileSpec,
    engine::{
        decision::{Targets, MAX_TARGETS},
        framework::{Picker, PoolWord, TieBand, DRAW_BOUND},
        view::{Candidate, ModelFingerprint, PoolFacts, RequestFacts},
    },
    policies::standard_filters,
    stages::{
        lifecycle::Lifecycle,
        pickers::{push_drawn, push_turn},
    },
    Draw,
};

/// How the `round_robin` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    ProfileSpec {
        filter: standard_filters(),
        score: vec![],
        pick: "round-robin".into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// A rotation per survivor set, keyed by the set's rows (id and DP rank) in
/// order; a set seen first starts from a row drawn from a shared counter. The
/// rotations live in a fixed table of 512 buckets the key picks one of, each
/// one cache line of eight words, each word holding a tag of its key and the
/// set's next position: a decision over a known set advances its word, a set
/// seen first claims an empty word of its bucket in place, and only a set
/// whose bucket is full takes a word, the one the counter names, so that
/// word's set draws anew when seen again (the forgetting a full table had,
/// per bucket: more than eight live sets of one bucket visited in a cycle
/// draw anew on every visit, a fresh row each rather than the next, which
/// under one table in a thousand sees at 500 live sets). No path takes a
/// lock or allocates; two sets racing for one word may take one position
/// from each other's rotation.
#[derive(Debug)]
pub struct RoundRobin {
    buckets: Box<[RotationBucket]>,
    /// Claims so far: a claim draws its start from this count and, in a
    /// full bucket, the word it takes.
    claims: AtomicU64,
}

/// One cache line of rotation words: a set lives in whichever word of its
/// bucket holds its tag.
#[derive(Debug)]
#[repr(align(64))]
struct RotationBucket {
    ways: [AtomicU64; WAYS],
}

/// Words per bucket: one cache line of them.
const WAYS: usize = 8;

/// Buckets of the rotation table: the top [`BUCKET_BITS`] of the mixed key
/// pick one, and the 31 bits below them are the word's tag.
const BUCKET_BITS: u32 = 9;

const ROTATION_BUCKETS: usize = 1 << BUCKET_BITS;

/// The odd multiplier that mixes a key before it is split into a bucket and
/// a tag, so two sets that differ in one row's rank (the key's top bits)
/// take different buckets.
const KEY_MIX: u64 = 0x9E37_79B9_7F4A_7C15;

/// A word no set has claimed: its tag has bit 31 set, which no key's tag has.
const EMPTY_SLOT: u64 = u64::MAX;

/// The low bits of a rotation word that hold the set's next position; the
/// tag sits above them. A position that wraps changes the tag, and the set
/// draws its start anew.
const POSITION_BITS: u32 = 32;

const POSITION_MASK: u64 = (1 << POSITION_BITS) - 1;

impl Default for RotationBucket {
    fn default() -> Self {
        Self {
            ways: [const { AtomicU64::new(EMPTY_SLOT) }; WAYS],
        }
    }
}

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            buckets: (0..ROTATION_BUCKETS)
                .map(|_| RotationBucket::default())
                .collect(),
            claims: AtomicU64::new(0),
        }
    }

    /// The set's identity: each row's id and DP rank in order, so two sets
    /// that differ by which ranks of a worker survive keep separate rotations.
    /// The rank rides the id's top bits, so a row costs one multiply as before
    /// (a second multiply per row cost the 512-row decision 7 %).
    pub(crate) fn set_key(candidates: &[Candidate], kept: &[u32]) -> u64 {
        Self::key_of(kept.iter().map(|&i| &candidates[i as usize]))
    }

    /// [`Self::set_key`] of every row of `rows` in their order: the key of
    /// a whole list. The list makes it once, and a whole pass that kept
    /// every row of the list turns the set's rotation by it with no row
    /// read ([`Picker::pick_among`]).
    #[cfg(feature = "in-process")]
    pub(crate) fn whole_key(rows: &[Candidate]) -> u64 {
        Self::key_of(rows.iter())
    }

    #[inline(always)]
    fn key_of<'a>(rows: impl Iterator<Item = &'a Candidate>) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for c in rows {
            h ^= c.id.0 ^ (u64::from(c.dp_rank) << 48);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    /// The key's bucket and its tag in place: the mixed key's top bits and
    /// the 31 below them, so every bit of the key reaches both.
    fn bucket_of(key: u64) -> (usize, u64) {
        let mixed = key.wrapping_mul(KEY_MIX);
        let bucket = (mixed >> (64 - BUCKET_BITS)) as usize;
        let tag = (mixed >> (64 - BUCKET_BITS - 31)) & 0x7FFF_FFFF;
        (bucket, tag << POSITION_BITS)
    }

    /// The position a claim starts from: the shared counter's value through
    /// the SplitMix64 finalizer, so sets seen first spread over the rows,
    /// and a set a full bucket forgets takes a fresh row at each claim
    /// rather than the row its cycle's length would fix. One multiply kept
    /// claims a cycle's length apart on rows a cycle's length apart, and a
    /// row count sharing a factor with it went unvisited.
    fn start_of(claim: u64) -> u64 {
        let mut z = claim.wrapping_add(KEY_MIX);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) >> 32
    }

    /// The key of a pool's own rotation, the one a decision that draws
    /// first turns where its host lends no word of the pool
    /// ([`PoolFacts::word`]): the model the pool serves and the request's
    /// phase, the pool's membership epoch, the list the host names
    /// ([`PoolFacts::reach`]) and its size. No row is read for it: by the
    /// host's word one list of rows stands behind one key until the pool's
    /// membership moves. Two lists behind one key would turn one rotation
    /// between them, and asks that alternate would draw every other row of
    /// each: a row one asker alone refuses would then be drawn for no one
    /// where that asker's turns land on it. The model's word is its
    /// fingerprint where the request carries it and its name's bytes where
    /// it does not: a host that carries it on some requests alone turns two
    /// rotations of the pool, each whole.
    fn pool_key(facts: &RequestFacts<'_>, pool: &PoolFacts<'_>, rows: usize) -> u64 {
        let mut key = FxHasher::default();
        if facts.model_fingerprint == ModelFingerprint::NONE {
            key.write(facts.model.as_bytes());
        } else {
            key.write_u64(facts.model_fingerprint.get());
        }
        key.write_u8(facts.phase as u8);
        key.write_u64(pool.membership_epoch);
        key.write_u64(pool.reach);
        key.write_usize(rows);
        key.finish()
    }

    fn advance(&self, key: u64) -> u64 {
        let (index, tag) = Self::bucket_of(key);
        let ways = &self.buckets[index].ways;
        loop {
            let mut empty = None;
            for way in ways {
                let word = way.load(Ordering::Relaxed);
                if word & !POSITION_MASK == tag {
                    return way.fetch_add(1, Ordering::Relaxed) & POSITION_MASK;
                }
                if word == EMPTY_SLOT && empty.is_none() {
                    empty = Some((way, word));
                }
            }
            // A set seen first claims an empty word at a drawn start; one
            // whose bucket is full takes the word the counter names, and that
            // word's set draws anew when seen. Another decision may have
            // changed the word meanwhile: the bucket is read again.
            let claim = self.claims.fetch_add(1, Ordering::Relaxed);
            let start = Self::start_of(claim);
            let (way, word) = empty.unwrap_or_else(|| {
                let way = &ways[(claim as usize) % WAYS];
                (way, way.load(Ordering::Relaxed))
            });
            if way
                .compare_exchange(
                    word,
                    tag | ((start + 1) & POSITION_MASK),
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                )
                .is_ok()
            {
                return start;
            }
        }
    }
}

impl Lifecycle for RoundRobin {}

impl Picker for RoundRobin {
    fn name(&self) -> &'static str {
        "round-robin"
    }

    // Picks by its rotation.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn draws_first(&self) -> bool {
        true
    }

    // Its rotation is its whole rule, and a pool's own word counts it.
    fn picks_alone(&self) -> bool {
        true
    }

    // Each pick takes its rotation's next turn.
    fn takes_turns(&self) -> bool {
        true
    }

    /// The pool's own turn by the pool's own word: one step of the word and
    /// one read of the worker the turn lands on, which is the pick where
    /// the caller keeps it. A turn that lands on a refused worker picks
    /// nobody: the caller reads the pool whole, and the workers that pass
    /// take turns by their own rotation ([`Picker::pick_among`]). The turn
    /// is not handed to the next worker of the list: a whole decision's
    /// turn is not either ([`Self::draw_first`]), so a list's requests take
    /// their workers by one rule. No table is looked up: the caller keeps
    /// the pool's rotation beside the pool.
    fn pick_alone(
        &self,
        pool: &PoolWord<'_>,
        keep: &mut dyn FnMut(usize) -> bool,
    ) -> Option<usize> {
        let rows = pool.rows;
        if rows == 0 {
            return None;
        }
        let turn = pool.word.fetch_add(1, Ordering::Relaxed) as usize % rows;
        keep(turn).then_some(turn)
    }

    /// The set's own turn, in the table every rotation of a set lives in.
    fn pick_among(&self, set: u64, passed: usize) -> usize {
        self.advance(set) as usize % passed.max(1)
    }

    /// The pool's own turn: the row the pool's rotation lands on is the pick
    /// when every filter keeps it, and the rows kept behind it follow as
    /// alternates, as far as [`DRAW_BOUND`] refused rows let the walk go. A
    /// turn that lands on a refused row is left to the whole pass, which
    /// turns the survivors' own rotation: over a turn of the pool each kept
    /// row takes its own turn and an even share of the refused rows'. With
    /// no row refused a turn of the pool visits every row once.
    ///
    /// The turn is counted in the pool's own word where the host lends it
    /// ([`PoolFacts::word`]), the word a pick alone counts in
    /// ([`Self::pick_alone`]): the list's own rotation, which no other list
    /// turns. Where none is lent the rotation is the one the pool's name
    /// finds: its model, the request's phase, its membership epoch, the
    /// list the host names and its size.
    ///
    /// A retry takes no turn of the pool: a request whose attempt is not its
    /// first ([`RequestFacts::attempt`]), or that names a worker it tried
    /// ([`RequestFacts::tried`]). The workers it may take are the ones it
    /// has not tried, and the turn is a first attempt's, whose workers are
    /// the list whole: the whole pass turns the rotation of the workers the
    /// retry is left. Either sign says so: a retry whose tried worker has
    /// left the list names none and states its number alone, and a host
    /// that states no number names the worker.
    fn draw_first(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
        _: &mut Draw,
    ) -> bool {
        let rows = candidates.len();
        if rows == 0 || facts.attempt != 0 || !facts.tried.is_empty() {
            return false;
        }
        let turn = match pool.word {
            Some(word) => word.fetch_add(1, Ordering::Relaxed),
            None => self.advance(Self::pool_key(facts, pool, rows)),
        };
        let pick = turn as usize % rows;
        if !keep(&candidates[pick]) {
            return false;
        }
        kept.push(pick as u32);
        let mut refused = 0;
        for step in 1..rows {
            if kept.len() == MAX_TARGETS || refused == DRAW_BOUND {
                break;
            }
            let row = (pick + step) % rows;
            if keep(&candidates[row]) {
                kept.push(row as u32);
            } else {
                refused += 1;
            }
        }
        true
    }

    fn order_drawn(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        push_drawn(candidates, kept, out);
    }

    fn pick(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        if kept.is_empty() {
            return;
        }
        let turn = self.advance(Self::set_key(candidates, kept)) as usize;
        push_turn(candidates, kept, turn, out);
    }
}

#[cfg(test)]
#[path = "../../tests/unit/policies/round_robin/tests.rs"]
mod tests;
