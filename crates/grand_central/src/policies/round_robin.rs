//! The `round_robin` policy: how it is composed, and the picker only it uses.

use std::{
    cell::Cell,
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
        pickers::{push_drawn, target_of},
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

/// How many pools a thread keeps a place for ([`row_of`]).
const KEPT_RUNS: usize = 4;

/// The lengths of a run of turns: a pool's runs take the first that does
/// not divide its size, so they start at each of its workers in turn. Runs
/// of a length that divided the size would start at few of its workers.
const RUN_LENGTHS: [u64; 2] = [61, 67];

/// The single steps a thread takes of a pool's word for each worker of the
/// pool before its first run of the word's turns ([`begin_run`]).
const STEPS_FIRST: u64 = 64;

/// The asks in a row that must find a held place unused before it goes to
/// the pool that asks ([`begin_run`]): a pool the thread still decides for
/// keeps its place beside a fifth one.
const IDLE_ASKS: u64 = 64;

/// A thread's place for one pool: its count of single steps, then its run
/// of turns of the pool's word.
struct Run {
    /// The pool: its word's address and its size, so one word asked under
    /// two sizes has a place for each. A pool of the same size made where
    /// a dropped pool's word stood keeps the place: it plays the turns the
    /// thread had left of that one first, as a run left behind, and takes
    /// the single steps that one had not taken.
    pool: Cell<(usize, usize)>,
    /// The run's next turn and the turn after its last: the same turn
    /// where the thread holds no run of the pool.
    next: Cell<u64>,
    end: Cell<u64>,
    /// The turn of the pool's first worker at or under the next turn: a
    /// turn's row is counted from it, and divided out once a lap.
    lap: Cell<u64>,
    /// The single steps the thread took of the pool's word in this place.
    steps: Cell<u64>,
    /// The next turn the place had when a pool last asked for it
    /// ([`begin_run`]).
    asked_at: Cell<u64>,
    /// The asks in a row that found no turn taken in the place since the
    /// ask before.
    idle: Cell<u64>,
}

impl Run {
    /// The row of the run's next turn, where the run has one left.
    #[inline]
    fn play(&self, rows: usize) -> Option<usize> {
        let turn = self.next.get();
        if turn == self.end.get() {
            return None;
        }
        self.next.set(turn.wrapping_add(1));
        let row = turn.wrapping_sub(self.lap.get());
        if row < rows as u64 {
            return Some(row as usize);
        }
        // The turn begins a lap of the pool.
        let row = turn % rows as u64;
        self.lap.set(turn - row);
        Some(row as usize)
    }

    /// One ask for the place: the asks in a row, this one counted, that
    /// found no turn taken in it since the ask before.
    fn ask(&self) -> u64 {
        let next = self.next.get();
        let idle = if self.asked_at.replace(next) == next {
            self.idle.get() + 1
        } else {
            0
        };
        self.idle.set(idle);
        idle
    }

    /// The two places change pools: each pool goes on in the other's place
    /// with all its own held, so no turn is dropped.
    fn swap(&self, other: &Self) {
        self.pool.swap(&other.pool);
        self.next.swap(&other.next);
        self.end.swap(&other.end);
        self.lap.swap(&other.lap);
        self.steps.swap(&other.steps);
        self.asked_at.swap(&other.asked_at);
        self.idle.swap(&other.idle);
    }
}

thread_local! {
    /// The calling thread's places, one a pool.
    static RUNS: [Run; KEPT_RUNS] = const {
        [const {
            Run {
                pool: Cell::new((0, 0)),
                next: Cell::new(0),
                end: Cell::new(0),
                lap: Cell::new(0),
                steps: Cell::new(0),
                asked_at: Cell::new(0),
                idle: Cell::new(0),
            }
        }; KEPT_RUNS]
    };
    /// The place a pool with none asks for next.
    static ASKED: Cell<usize> = const { Cell::new(0) };
}

/// The row the calling thread's next turn lands on, of the pool of `rows`
/// workers whose word is `word`: the row a single step of the word gives
/// that turn.
///
/// A thread that stays with a pool takes a run of the word's turns at once
/// and plays them one by one, so the word every deciding thread shares is
/// written once a run ([`begin_run`]). Each turn of the word is handed out
/// once and the turns of a run follow one another: a thread that decides
/// alone takes the pool's workers in the order single steps give for as
/// long as it keeps the pool's place. Across threads the picks follow no
/// one order. Until the word wraps, for one pool, a word under one size,
/// the turns two workers took part by one at most, and by one more for
/// each run that holds turns not played: the run a thread plays from, and
/// each run left behind, by a thread that ended or by a place given up
/// ([`begin_run`]).
#[inline]
fn row_of(word: &AtomicU64, rows: usize) -> usize {
    first_turn(word, rows).unwrap_or_else(|| later_turn(word, rows))
}

/// The pool's row where the thread's first place is the pool's and holds a
/// turn: all a thread that decides for one pool reads.
#[inline]
fn first_turn(word: &AtomicU64, rows: usize) -> Option<usize> {
    let pool = (std::ptr::from_ref(word).addr(), rows);
    RUNS.with(|runs| {
        let first = &runs[0];
        if first.pool.get() == pool {
            first.play(rows)
        } else {
            None
        }
    })
}

/// The pool's row where the thread's first place is not the pool's or holds
/// no turn: the next turn of the pool's own place, or a turn the thread
/// takes of the word. Kept out of line, so a thread that decides for one
/// pool pays for the one place it reads.
#[inline(never)]
fn later_turn(word: &AtomicU64, rows: usize) -> usize {
    let pool = (std::ptr::from_ref(word).addr(), rows);
    RUNS.with(|runs| {
        let own = runs.iter().find(|run| run.pool.get() == pool);
        own.and_then(|run| run.play(rows))
            .unwrap_or_else(|| begin_run(runs, own, word, pool))
    })
}

/// [`Picker::pick_alone`] where the thread's first place gives no turn.
/// Kept out of line whole, with its call of `keep`, so the pick the first
/// place gives saves no register around this one.
#[inline(never)]
fn later_pick(pool: &PoolWord<'_>, keep: &mut dyn FnMut(usize) -> bool) -> Option<usize> {
    let turn = later_turn(pool.word, pool.rows);
    keep(turn).then_some(turn)
}

/// The row of a turn the calling thread takes of the pool's word: a single
/// step, or the first turn of a run.
///
/// A thread's first turns of a pool are single steps, [`STEPS_FIRST`] for
/// each worker of the pool: a thread that ends before leaves no turn
/// unplayed. Past them it takes a run whenever it holds none, and a thread
/// that ends then leaves the turns of one run, which part the picks of two
/// workers by one at most. A run stops at the top of the word: the turns
/// it took past the top are not played, so no row is counted across the
/// wrap.
///
/// The steps are counted in the pool's place. A pool with none asks the
/// thread's places for one, one place a turn: a place no pool holds gives
/// itself at once; a held place gives itself up once [`IDLE_ASKS`] asks in
/// a row found no turn taken in it, and the turns its run had left are not
/// played; a place in use stays. A pool takes single steps for as
/// long as it has no place, so a fifth pool beside four in use pays this
/// look and a step of the word on every turn.
///
/// A pick reads the thread's first place in line ([`first_turn`]). A pool
/// that takes a run from a later place asks for the first: where that gave
/// no turn since the ask before, the two pools change places, each with
/// all its place held. So a pool the thread decides for alone, as one made
/// anew where its workers changed, is read in line from its second run on.
#[cold]
fn begin_run(
    runs: &[Run; KEPT_RUNS],
    own: Option<&Run>,
    word: &AtomicU64,
    pool: (usize, usize),
) -> usize {
    let rows = pool.1 as u64;
    let place = own.or_else(|| {
        let run = &runs[ASKED.replace((ASKED.get() + 1) % KEPT_RUNS)];
        (run.ask() >= IDLE_ASKS || run.pool.get().1 == 0).then(|| {
            run.pool.set(pool);
            run.steps.set(0);
            run.idle.set(0);
            run
        })
    });
    let Some(mut run) = place else {
        return (word.fetch_add(1, Ordering::Relaxed) % rows) as usize;
    };
    let steps = run.steps.get();
    let length = if steps < STEPS_FIRST.saturating_mul(rows) {
        run.steps.set(steps + 1);
        1
    } else {
        RUN_LENGTHS
            .into_iter()
            .find(|&length| !rows.is_multiple_of(length))
            .unwrap_or(1)
    };
    let first_place = &runs[0];
    if length > 1 && !std::ptr::eq(run, first_place) && first_place.ask() > 0 {
        run.swap(first_place);
        run = first_place;
    }
    let first = word.fetch_add(length, Ordering::Relaxed);
    let row = first % rows;
    run.next.set(first.wrapping_add(1));
    run.end.set(first.checked_add(length).unwrap_or(0));
    run.lap.set(first - row);
    run.asked_at.set(u64::MAX);
    row as usize
}

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
    fn set_key(candidates: &[Candidate], kept: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &i in kept {
            let c = &candidates[i as usize];
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

    /// The pool's own turn by the pool's own word: one turn of the word and
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
        let Some(turn) = first_turn(pool.word, rows) else {
            return later_pick(pool, keep);
        };
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
        let pick = match pool.word {
            Some(word) => row_of(word, rows),
            None => self.advance(Self::pool_key(facts, pool, rows)) as usize % rows,
        };
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
        let position = self.advance(Self::set_key(candidates, kept)) as usize;
        for k in 0..MAX_TARGETS.min(kept.len()) {
            out.push(target_of(
                &candidates[kept[(position + k) % kept.len()] as usize],
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stages::scorers::tests::{candidate, kept};

    fn first_pick(picker: &dyn Picker, facts: &RequestFacts<'_>, pool: &[Candidate]) -> u32 {
        let mut out = Targets::new();
        let kept = kept(pool.len() as u32);
        let scores = vec![0.0; pool.len()];
        picker.pick(facts, pool, &kept, &scores, TieBand::NORMALIZED, &mut out);
        out.first().map(|t| t.position).unwrap_or(u32::MAX)
    }

    #[test]
    fn round_robin_walks_a_set_in_turn_and_keeps_sets_apart() {
        let rr = RoundRobin::new();
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let facts = RequestFacts::default();
        let seen: Vec<u32> = (0..6).map(|_| first_pick(&rr, &facts, &pool)).collect();
        let start = seen[0];
        for (k, &p) in seen.iter().enumerate() {
            assert_eq!(p, (start as usize + k) as u32 % 3, "{seen:?}");
        }
        // Another set has its own rotation.
        let mut out = Targets::new();
        rr.pick(
            &facts,
            &pool,
            &[0, 1],
            &[0.0; 2],
            TieBand::NORMALIZED,
            &mut out,
        );
        assert!(out.first().is_some());
    }

    /// A draw counts its turn in the pool's own word where the host lends
    /// one, and claims no rotation by the pool's name. Where none is lent
    /// the named rotation turns and the word stands.
    #[test]
    fn a_draw_turns_the_lent_word_and_names_no_rotation() {
        let rr = RoundRobin::new();
        let pool: Vec<Candidate> = (0..16).map(|row| candidate(row, 0)).collect();
        let facts = RequestFacts::default();
        let word = AtomicU64::new(21);
        let lent = PoolFacts {
            word: Some(&word),
            ..PoolFacts::default()
        };
        let drawn = |pool_facts: &PoolFacts<'_>| {
            let mut kept = Vec::new();
            let mut draw = Draw::new(1);
            let mut all = |_: &Candidate| true;
            assert!(rr.draw_first(&facts, pool_facts, &pool, &mut all, &mut kept, &mut draw));
            kept
        };
        // The word's turn, 21, lands on row 5 of 16, and the next on row 6.
        assert_eq!(drawn(&lent), [5, 6, 7, 8]);
        assert_eq!(drawn(&lent), [6, 7, 8, 9]);
        let read = |word: &AtomicU64| word.load(Ordering::Relaxed);
        assert_eq!((read(&word), read(&rr.claims)), (23, 0));
        drawn(&PoolFacts::default());
        assert_eq!((read(&word), read(&rr.claims)), (23, 1));
    }

    /// A thread's first turns of a pool are single steps of the pool's
    /// word, so threads that take one turn and end leave every turn played.
    /// A thread that stays takes the word's turns a run at a time, in the
    /// order single steps give.
    #[test]
    fn a_thread_takes_a_pools_turns_in_runs_once_it_stays() {
        // On a thread of its own, whose places are all free.
        let stays = std::thread::spawn(|| {
            let rr = RoundRobin::new();
            let word = AtomicU64::new(0);
            let pool = PoolWord {
                rows: 62,
                word: &word,
            };
            let turn = || rr.pick_alone(&pool, &mut |_| true).expect("a kept row");
            // Threads that take one turn of the pool and end, a turn of
            // this thread between them: every worker takes one.
            let mut picked = [0; 62];
            for _ in 0..31 {
                let once = std::thread::scope(|ends| ends.spawn(turn).join());
                picked[once.expect("its turn")] += 1;
                picked[turn()] += 1;
            }
            assert_eq!(picked, [1; 62]);
            // This thread stays. Its single steps, 64 a worker, end at the
            // turn 3,999: from there one write of the word takes 61 turns,
            // and the thread's rows go on in the order single steps give.
            let steps = 31 + STEPS_FIRST as usize * 62;
            for step in 62..steps {
                assert_eq!(turn(), step % 62);
            }
            assert_eq!(word.load(Ordering::Relaxed), steps as u64);
            for played in steps..steps + 62 {
                assert_eq!(turn(), played % 62);
            }
            assert_eq!(word.load(Ordering::Relaxed), steps as u64 + 2 * 61);
            // A run stops at the top of the word. The thread plays the 60
            // turns left of its run, then takes 61 turns 31 under the top:
            // it plays those 31, and its next run starts at the turn 30,
            // where the word stands.
            word.store(u64::MAX - 30, Ordering::Relaxed);
            for _ in 0..60 {
                turn();
            }
            for under in (0..31).rev() {
                assert_eq!(turn() as u64, (u64::MAX - under) % 62);
            }
            assert_eq!((turn(), word.load(Ordering::Relaxed)), (30, 30 + 61));
        });
        stays.join().expect("the thread's asserts hold");
    }

    /// A fifth pool takes single steps beside four that are decided for:
    /// none of the four loses its place to it. Decided for alone, it takes
    /// the place of one that went unused and moves to the thread's first.
    #[test]
    fn a_fifth_pool_leaves_four_pools_in_use_their_places() {
        let places = std::thread::spawn(|| {
            let rr = RoundRobin::new();
            let words: [AtomicU64; 5] = std::array::from_fn(|_| AtomicU64::new(0));
            let pick = |list: usize| {
                let pool = PoolWord {
                    rows: 3,
                    word: &words[list],
                };
                rr.pick_alone(&pool, &mut |_| true).expect("a kept row")
            };
            // Four pools past their single steps, 64 for each of 3 workers.
            for step in 0..192 * 4 {
                pick(step % 4);
            }
            // Five pools in an order that leaves none out for long: the
            // four go on in runs of 61, the fifth in single steps.
            let (mut order, mut fifth) = (1_u64, 0);
            for _ in 0..10_000 {
                order = order
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let list = (order >> 33) as usize % 5;
                fifth += u64::from(list == 4);
                pick(list);
            }
            for word in &words[..4] {
                assert_eq!((word.load(Ordering::Relaxed) - 192) % 61, 0);
            }
            assert_eq!(words[4].load(Ordering::Relaxed), fifth);
            // The fifth decided for alone: a place goes to it once 64 asks
            // in a row found it unused, and past its own single steps it
            // goes on in runs, read from the thread's first place.
            for _ in 0..4 * 65 + 192 + 61 {
                pick(4);
            }
            let before = words[4].load(Ordering::Relaxed);
            for _ in 0..60 {
                pick(4);
            }
            assert_eq!((words[4].load(Ordering::Relaxed) - before) % 61, 0);
            let first = RUNS.with(|runs| runs[0].pool.get());
            assert_eq!(first, (std::ptr::from_ref(&words[4]).addr(), 3));
        });
        places.join().expect("the thread's asserts hold");
    }

    #[test]
    fn two_sets_of_one_bucket_keep_their_own_rotations() {
        // Keys a and b share a bucket under different tags. A fresh table's
        // counter runs 0, 1, 2, ... per claim: a starts at the row drawn from
        // 0 and b from 1, and seen in turn for a thousand rounds each walks
        // on by one, neither taking the other's word.
        let rr = RoundRobin::new();
        let a = 0x0123_4567_89ab_cdefu64;
        let (bucket_a, tag_a) = RoundRobin::bucket_of(a);
        let b = (1u64..)
            .find(|&k| k != a && RoundRobin::bucket_of(k).0 == bucket_a)
            .expect("a key sharing a's bucket");
        assert_ne!(RoundRobin::bucket_of(b).1, tag_a, "b's tag differs");
        let (start_a, start_b) = (RoundRobin::start_of(0), RoundRobin::start_of(1));
        for n in 0..1_000 {
            assert_eq!(
                [rr.advance(a), rr.advance(b)],
                [start_a + n, start_b + n],
                "round {n}"
            );
        }
    }

    #[test]
    fn a_ninth_set_of_a_full_bucket_takes_the_word_the_counter_names() {
        // Nine keys share a bucket of eight words. The ninth, claim 8, takes
        // word 0 from the first set; the first, seen again, claims at 9 and
        // takes word 1 from the second; a set whose word nobody took walks
        // on. Each claim starts at the row drawn from its counter value.
        let rr = RoundRobin::new();
        let bucket = RoundRobin::bucket_of(0x0123_4567_89ab_cdef).0;
        let keys: Vec<u64> = (1u64..)
            .filter(|&k| RoundRobin::bucket_of(k).0 == bucket)
            .take(9)
            .collect();
        let start = RoundRobin::start_of;
        for (n, &k) in keys[..8].iter().enumerate() {
            assert_eq!(
                rr.advance(k),
                start(n as u64),
                "each claims a word of its own"
            );
        }
        assert_eq!(rr.advance(keys[8]), start(8), "the ninth takes word 0");
        assert_eq!(rr.advance(keys[2]), start(2) + 1, "the third walks on");
        assert_eq!(
            rr.advance(keys[0]),
            start(9),
            "the first draws anew, at word 1"
        );
        assert_eq!(rr.advance(keys[8]), start(8) + 1, "the ninth walks on");
        assert_eq!(rr.advance(keys[1]), start(10), "the second draws anew");
    }

    #[test]
    fn nine_sets_cycling_one_bucket_take_every_row() {
        // Nine keys of one bucket visited in a fixed cycle: eight words cannot
        // hold them, so every visit past the first eight draws anew, at the
        // row its counter value draws. Reduced to 2, 3, 4, 9, 12 or 14 rows,
        // every set takes every row within a third of its even share over a
        // thousand rounds, where a start that advanced by the cycle's length
        // fixed one row, and one multiply left rows sharing a factor with
        // the cycle's stride unvisited.
        let rr = RoundRobin::new();
        let bucket = RoundRobin::bucket_of(0x0123_4567_89ab_cdef).0;
        let keys: Vec<u64> = (1u64..)
            .filter(|&k| RoundRobin::bucket_of(k).0 == bucket)
            .take(9)
            .collect();
        const ROWS: [usize; 6] = [2, 3, 4, 9, 12, 14];
        let rounds = 1_000u64;
        let mut taken = vec![[[0u64; 14]; ROWS.len()]; 9];
        for round in 0..rounds {
            for (n, &k) in keys.iter().enumerate() {
                let position = rr.advance(k);
                assert_eq!(
                    position,
                    RoundRobin::start_of(9 * round + n as u64),
                    "round {round}, set {n}"
                );
                for (r, &rows) in ROWS.iter().enumerate() {
                    taken[n][r][(position % rows as u64) as usize] += 1;
                }
            }
        }
        for (n, set) in taken.iter().enumerate() {
            for (r, &rows) in ROWS.iter().enumerate() {
                for (row, &count) in set[r][..rows].iter().enumerate() {
                    assert!(
                        count * rows as u64 * 3 >= rounds,
                        "set {n} over {rows} rows: row {row} taken {count} of {rounds}"
                    );
                }
            }
        }
    }

    #[test]
    fn survivor_sets_that_differ_by_rank_alone_rotate_apart() {
        // A is a two-rank worker whose rows share its id; a per-rank bit drops
        // one rank at a time, so the survivors alternate between [A1, B] and
        // [A0, B]. Each set keeps its own rotation and walks both of its rows.
        let rr = RoundRobin::new();
        let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        pool[1].id = pool[0].id;
        pool[1].dp_rank = 1;
        let x = [1u32, 2];
        let y = [0u32, 2];
        assert_ne!(
            RoundRobin::set_key(&pool, &x),
            RoundRobin::set_key(&pool, &y)
        );
        let facts = RequestFacts::default();
        let mut seen_x = [0usize; 3];
        let mut seen_y = [0usize; 3];
        for _ in 0..4 {
            for (kept, seen) in [(&x[..], &mut seen_x), (&y[..], &mut seen_y)] {
                let mut out = Targets::new();
                rr.pick(
                    &facts,
                    &pool,
                    kept,
                    &[0.0; 2],
                    TieBand::NORMALIZED,
                    &mut out,
                );
                seen[out.first().expect("a pick").position as usize] += 1;
            }
        }
        assert_eq!(seen_x, [0, 2, 2], "X walks A1 and B in turn: {seen_x:?}");
        assert_eq!(seen_y, [2, 0, 2], "Y walks A0 and B in turn: {seen_y:?}");
    }
}
