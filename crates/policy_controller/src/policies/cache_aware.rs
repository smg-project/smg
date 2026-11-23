//! The `cache_aware` policy: how it is composed, and the affinity stages
//! written for it.
//!
//! The affinity stages of a profile: `prefix-affinity` (the holders of the
//! request's prefix, with the match threshold and the spill gate),
//! `warmup-slice` (the slice of misses and the diverted hits a worker with a
//! thin or young cache takes) and `kv-pressure` (the fleet gate under which
//! affinity yields to load). The three read one set of [`Gates`], so a
//! profile's spill and KV-pressure parameters configure all of them: the
//! `kv-pressure` stage orders nothing itself, and the spill gate the slice
//! checks its candidates against is the affinity stage's. What they read of
//! a candidate beyond its row (the request's blocks it holds, its age in the
//! pool, its index size) is the affinity column the host fills beside the
//! rows ([`RequestFacts::affinity`]) or the rows it names in the column's
//! place ([`AffinityRows`]), never the row. A stateful part (the
//! slice's period, the diversion's window per worker) keeps its state behind
//! the lifecycle alone, and no stage allocates per decision.
//!
//! The affinity score is a rank, not a share: the candidates holding the
//! deepest match of the request and under the spill gate come first, the
//! rows under the gate next, the rest last, so the profile's load scorer
//! orders within the group as the expected wait orders among equal holders
//! today. A share-weighted sum cannot hold a deeper holder ahead of a lighter
//! one, which is the rule the preset keeps.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    document::spec::{
        AffinitySource, BestParams, KvPressureParams, Needs, PickStage, PrefixAffinityParams,
        ProfileSpec, ScoreStage, WarmupSliceParams,
    },
    engine::{
        decision::Overlap,
        framework::{ScoreFront, ScoreSpread, Scorer, TieBand},
        prefix_pass::{keep_prefix_pass, take_prefix_pass, withdraw_prefix_pass, PrefixStageId},
        view::{
            blocks_within, mean_in_flight, units_within, AffinityFacts, AffinityRows, Candidate,
            Feed, HashKind, LoadRow, Phase, RequestFacts, Walk, WorkerId, TEXT_BYTES_PER_TOKEN,
        },
        warm_up::WarmUpRule,
    },
    policies::standard_filters,
    stages::{lifecycle::Lifecycle, scorers::ReadsLoad},
    tracking::{in_flight::with_counts, lease::Membership},
    Draw,
};

/// How the `cache_aware` policy is composed: its filters, its scorers and its
/// picker.
pub(crate) fn profile() -> ProfileSpec {
    // Affinity first, the expected wait within the group: the slice
    // and the diversion over everything when due, the holder tiers
    // over the load, the fleet gate configuring both. The weights
    // are ranks: each stage's values are 0, 0.5 or 1, so a stage
    // outweighs everything after it.
    ProfileSpec {
        filter: standard_filters(),
        score: vec![
            ScoreStage::WarmupSlice(WarmupSliceParams {
                weight: 8.0,
                ..WarmupSliceParams::default()
            })
            .into(),
            ScoreStage::PrefixAffinity(PrefixAffinityParams {
                weight: 4.0,
                ..PrefixAffinityParams::default()
            })
            .into(),
            "kv-pressure".into(),
            "least-load".into(),
        ],
        // A large pool is decided by a draw of sixteen rows and the
        // rows the host names: its holders and its warming rows.
        pick: PickStage::Best(BestParams { sample: 16 }).into(),
        sent_ttl_ms: None,
        sent_report_delay_ms: None,
    }
}

/// The floor of the spill gate's swing-sized absolute margin: a handful of
/// requests above the mean on an otherwise idle pool is not pressure.
const MIN_ABSOLUTE_MARGIN: f32 = 4.0;

/// A match of at most this many blocks counts as a miss for the slice when
/// it is under half the request: a chat template's head, shared by every
/// holder, which costs less to recompute than keeping a warming worker idle.
const WARMUP_MISS_BLOCKS: u32 = 4;

/// A hit whose deepest match is at most this share of the request is a
/// shallow one: diverting it to a thin worker recomputes little, so these
/// go first.
const DIVERT_SHALLOW_SHARE: f32 = 0.5;

/// A thin worker with requests in flight takes no second diverted hit
/// sooner than this after its last one.
const DIVERT_WINDOW_MICROS: u64 = 2_000_000;

/// Tokens per block when the document names none: the block size of record
/// of the common engines.
const DEFAULT_BLOCK_SIZE: u32 = 16;

/// The matched share under which a request is balanced instead of pinned,
/// when the document names no threshold. The warm-up slice counts a hit as
/// a miss at or under the same share when it names none.
pub const DEFAULT_THRESHOLD: f32 = 0.3;

/// How long after admission a worker counts as warming, in seconds, when
/// the document names no time.
pub const DEFAULT_WARMUP_SECS: u64 = 60;

/// The share of misses offered to warming workers when the document names
/// none.
pub const DEFAULT_WARMUP_SHARE: f32 = 0.25;

/// The index growth after which a worker is warm, in blocks, when the
/// document names none.
pub const DEFAULT_WARMUP_BLOCKS: u64 = 1024;

/// The share of the fleet's median index under which a worker is thin, when
/// the document names none.
pub const DEFAULT_WARMUP_THIN_RATIO: f32 = 0.5;

/// One hit in this many is diverted to a thin worker when the document
/// names no count.
pub const DEFAULT_WARMUP_DIVERT_EVERY: u64 = 8;

/// The gates the affinity stages of one profile share: the spill gate of
/// `prefix-affinity`, the fleet KV gate of `kv-pressure`, and the block size
/// a held prefix is priced by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    /// A holder loaded past this multiple of the mean in flight (the kept
    /// rows', or the drawn rows' behind a draw) AND past the absolute
    /// margin above it spills.
    pub rel: f32,
    /// The absolute margin's configured part, in requests; the margin is
    /// capped at the larger of the count's natural swing at the mean and
    /// the held prefix's price, and never under four unless this is lower.
    pub abs: u32,
    /// Tokens per request a held prefix is priced at in the margin; `0`
    /// leaves the margin to the swing.
    pub prefix_margin_tokens: u32,
    /// The usage spread (hottest minus coldest, as a ratio) past which
    /// affinity yields to load; `1.0` and above is off.
    pub kv_spread: f32,
    /// The usage the hottest candidate crosses for affinity to yield to
    /// load; `1.0` and above is off.
    pub kv_ceiling: f32,
    /// Tokens per block.
    pub block_size: u32,
}

impl Default for Gates {
    fn default() -> Self {
        Self {
            rel: 1.5,
            abs: 64,
            prefix_margin_tokens: 0,
            kv_spread: 1.0,
            kv_ceiling: 1.0,
            block_size: DEFAULT_BLOCK_SIZE,
        }
    }
}

impl Gates {
    /// The gates of a profile from the parameters its affinity stages wrote
    /// and the view's block size.
    pub fn of(
        affinity: Option<&PrefixAffinityParams>,
        kv: Option<&KvPressureParams>,
        block_size: Option<u32>,
    ) -> Self {
        let defaults = Self::default();
        let spill = affinity.and_then(|p| p.spill.as_ref());
        Self {
            rel: spill.and_then(|s| s.rel).unwrap_or(defaults.rel),
            abs: spill.and_then(|s| s.abs).unwrap_or(defaults.abs),
            prefix_margin_tokens: spill
                .and_then(|s| s.prefix_margin_tokens)
                .unwrap_or(defaults.prefix_margin_tokens),
            kv_spread: kv.and_then(|k| k.spread).unwrap_or(defaults.kv_spread),
            kv_ceiling: kv.and_then(|k| k.ceiling).unwrap_or(defaults.kv_ceiling),
            block_size: block_size.unwrap_or(defaults.block_size).max(1),
        }
    }

    /// Whether the KV gate reads the fleet at all.
    fn kv_configured(&self) -> bool {
        self.kv_spread < 1.0 || self.kv_ceiling < 1.0
    }

    /// Whether backend KV pressure demands abandoning affinity: the hottest
    /// kept candidate over the ceiling, or materially hotter than the
    /// coldest. Read from the active share of the rows that report; a pool
    /// without a report is never under pressure.
    fn kv_imbalanced(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> bool {
        if !self.kv_configured() {
            return false;
        }
        let mut bounds: Option<(f32, f32)> = None;
        for &i in kept {
            let Some(load) = load(&candidates[i as usize]) else {
                continue;
            };
            let usage = load.active_kv();
            bounds = Some(bounds.map_or((usage, usage), |(lo, hi)| (lo.min(usage), hi.max(usage))));
        }
        let Some((lo, hi)) = bounds else {
            return false;
        };
        hi > self.kv_ceiling || hi - lo > self.kv_spread
    }

    /// The absolute margin at `avg` for a candidate holding `held_tokens` of
    /// the request: the configured part, capped at the larger of the
    /// count's natural swing (two standard deviations of a Poisson count,
    /// rounded up) and the held prefix's price, never under the floor unless
    /// the configured part is lower.
    fn margin(&self, avg: f32, held_tokens: f32) -> f32 {
        let swing = (2.0 * avg.sqrt()).ceil();
        let price = if self.prefix_margin_tokens == 0 || held_tokens <= 0.0 {
            0.0
        } else {
            (held_tokens / self.prefix_margin_tokens as f32).ceil()
        };
        (self.abs as f32).min(swing.max(MIN_ABSOLUTE_MARGIN).max(price))
    }

    /// Whether a candidate with `load` requests in flight is over the spill
    /// gate at the mean `avg`.
    fn over(&self, load: f32, avg: f32, held_tokens: f32) -> bool {
        load > avg * self.rel && load > avg + self.margin(avg, held_tokens)
    }

    /// The spill gate at the mean `avg` for a candidate holding
    /// `held_tokens` of the request, as [`Self::over`] tests it: the two
    /// bounds a pass behind a draw takes once.
    fn spill(&self, avg: f32, held_tokens: f32) -> Spill {
        Spill {
            rel: avg * self.rel,
            abs: avg + self.margin(avg, held_tokens),
        }
    }

    /// The size of one of the request's blocks: the facts' own when the
    /// host counted with one (an index's learned size), else the view's.
    fn block_size_of(&self, facts: &RequestFacts<'_>) -> u32 {
        facts.block_size_or(self.block_size)
    }

    /// How many whole blocks of the request lie within its first `tokens`
    /// tokens ([`blocks_within`]), at the block size the request was
    /// counted with.
    pub(crate) fn whole_blocks(&self, tokens: u32, facts: &RequestFacts<'_>) -> u32 {
        blocks_within(tokens, facts.hash_kind, self.block_size_of(facts))
    }

    /// The request's size in blocks ([`RequestFacts::blocks`]): its token
    /// count over the block size (the text kind's bytes over the window's
    /// size in bytes, as `tokens_of` reads them back), never under one and
    /// never under the block hashes the host sent.
    fn request_blocks(&self, facts: &RequestFacts<'_>) -> u32 {
        facts.blocks(self.block_size)
    }

    /// What a sent prefix's share of the request is taken of
    /// ([`Whole::clears`]): one of the request's blocks and the request's
    /// size, both in the unit the request was counted in ([`units_within`]).
    /// The size is the whole prompt's, not its whole blocks': a prompt that
    /// ends inside a block is larger than they say. It is never under one
    /// block and never under the block hashes the host sent, as
    /// [`Self::request_blocks`] holds its count.
    fn whole(&self, facts: &RequestFacts<'_>) -> Whole {
        let block = self.block_size_of(facts);
        let hashed = u32::try_from(facts.block_hashes.len()).unwrap_or(u32::MAX);
        let size = units_within(facts.prompt_size(), facts.hash_kind);
        Whole {
            block,
            size: size.max(block).max(hashed.saturating_mul(block)),
        }
    }

    /// `blocks` of the request in tokens, for the held prefix's price: at
    /// the block size the request was counted with.
    fn tokens_of(&self, blocks: u32, facts: &RequestFacts<'_>) -> f32 {
        let units = blocks as f32 * self.block_size_of(facts) as f32;
        match facts.hash_kind {
            HashKind::Token => units,
            HashKind::Text => units / TEXT_BYTES_PER_TOKEN as f32,
        }
    }
}

/// A request's size and the size of one of its blocks, in one unit
/// ([`Gates::whole`]).
#[derive(Clone, Copy, Debug)]
struct Whole {
    block: u32,
    size: u32,
}

impl Whole {
    /// Whether `blocks` blocks sent to a worker are more than the share
    /// `threshold` of the request. The share is one division of two whole
    /// numbers, the blocks' size over the request's: the two the classic
    /// policy divides, its matched tokens over the prompt's. Blocks over a
    /// count of blocks would round the request's size first.
    fn clears(self, blocks: u32, threshold: f32) -> bool {
        blocks.saturating_mul(self.block) as f32 / self.size as f32 > threshold
    }
}

/// The spill gate at one mean and one held prefix ([`Gates::spill`]).
#[derive(Clone, Copy, Debug)]
struct Spill {
    rel: f32,
    abs: f32,
}

impl Spill {
    /// Whether `load` requests in flight are over the gate, as
    /// [`Gates::over`] answers. Both bounds are tested whatever the first
    /// says: over rows a draw chose, a test that stops at the first is a
    /// branch nothing predicts.
    #[inline]
    fn over(self, load: f32) -> bool {
        (load > self.rel) & (load > self.abs)
    }
}

/// How a pass of the warm-up stage tells a warming row: by the stage's own
/// rule over a host's column, or by the rows the host named.
#[derive(Clone, Copy)]
enum Warm<'a> {
    /// Over a column: the fleet's index level, whether any row is thin
    /// against it, and whether every row is young.
    Ruled {
        level: u32,
        thin_any: bool,
        young_all: bool,
    },
    /// The host's warming rows, ascending, and whether they are the thin.
    Named { rows: &'a [u32], thin: bool },
}

impl Warm<'_> {
    /// Whether the warming rows are thin ones: a hit is diverted to no
    /// other.
    fn thin(self) -> bool {
        match self {
            Self::Ruled { thin_any, .. } => thin_any,
            Self::Named { thin, .. } => thin,
        }
    }
}

/// What one pass of the warm-up stage over a decision reads: the rows, what
/// they hold, how a warming row is told, the request's deepest index match
/// and size, and the decision's draw.
struct Pass<'a> {
    facts: &'a RequestFacts<'a>,
    candidates: &'a [Candidate],
    kept: &'a [u32],
    feed: Feed<'a>,
    warm: Warm<'a>,
    best: u32,
    request_blocks: u32,
    draw: Draw,
}

impl Pass<'_> {
    /// The slot of the least-loaded kept row `wanted` admits, equal loads
    /// drawn uniformly from the decision's seed. `wanted` is asked with
    /// the row's requests in flight, the decision's one number for the
    /// row.
    fn least_loaded(
        &mut self,
        mut wanted: impl FnMut(u32, &Candidate, u32) -> bool,
    ) -> Option<usize> {
        let mut pick: Option<(usize, u32)> = None;
        let mut tied = 0usize;
        with_counts!(self.facts, |counts| {
            for (slot, &i) in self.kept.iter().enumerate() {
                let c = &self.candidates[i as usize];
                let held = counts(c);
                if !wanted(i, c, held) {
                    continue;
                }
                match pick {
                    Some((_, load)) if held > load => {}
                    Some((_, load)) if held == load => {
                        tied += 1;
                        if self.draw.below(tied + 1) == 0 {
                            pick = Some((slot, load));
                        }
                    }
                    _ => {
                        pick = Some((slot, held));
                        tied = 0;
                    }
                }
            }
        });
        pick.map(|(slot, _)| slot)
    }
}

/// The mean in flight the spill gate reads: the drawn rows' where the
/// engine set it
/// ([`AffinityRows::mean_in_flight`]),
/// else the kept rows'.
// Forced into the stage's passes with the mean it reads
// ([`mean_in_flight`]): out of line the mean's walk costs a pass a call.
#[inline(always)]
fn spill_mean(
    drawn: Option<f32>,
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
) -> f32 {
    drawn.unwrap_or_else(|| mean_in_flight(facts, candidates, kept))
}

/// Every row scores alike: the stage has nothing to say about this request.
fn neutral(out: &mut [f32]) -> TieBand {
    out.fill(1.0);
    TieBand(0.0)
}

// ---------------------------------------------------------------- prefix affinity

/// The holders of the request's prefix first. A candidate's match is its
/// index overlap at any depth and, when the source reads placements too, a
/// predicted placement whose share of the request clears the threshold. The
/// share is of the whole prompt, as the classic policy takes it: the size
/// of the placement's blocks over the prompt's size. The rows holding the
/// deepest match and under the spill gate score `1.0`, the rows under the
/// gate `0.5`, the rest `0.0`; under the fleet KV gate every row scores
/// alike and the load scorer decides. The values are exact, so the band is
/// zero.
#[derive(Debug)]
pub struct PrefixAffinity {
    threshold: f32,
    source: AffinitySource,
    gates: Gates,
    /// The stage's own number, by which what its pass measured is told from
    /// another stage's.
    id: PrefixStageId,
}

impl PrefixAffinity {
    /// The stage from its parameters and its profile's gates.
    pub fn new(params: &PrefixAffinityParams, gates: Gates) -> Self {
        Self {
            threshold: params.threshold.unwrap_or(DEFAULT_THRESHOLD),
            source: params.source.unwrap_or_default(),
            gates,
            id: PrefixStageId::next(),
        }
    }

    /// Whether a held prefix is priced for this request: the document
    /// prices the margin, and a decode worker prefills nothing again.
    fn priced(&self, facts: &RequestFacts<'_>) -> bool {
        self.gates.prefix_margin_tokens > 0 && facts.phase != Phase::Decode
    }

    /// Leaves what this pass judges by for the word of the pick
    /// ([`keep_prefix_pass`]). The first stage to judge on a thread keeps
    /// the slot, and a later pass of its own writes over it.
    fn judge(&self, best: u32, mean: f32) {
        keep_prefix_pass(self.id, best, mean);
    }

    /// This pass orders nothing: what an earlier pass of this stage left
    /// is no word of the pick.
    fn unjudge(&self) {
        withdraw_prefix_pass(self.id);
    }

    /// The blocks of the request a candidate holds by this stage's source,
    /// `whole` being what a predicted placement's share is taken of
    /// ([`Gates::whole`]).
    fn matched(&self, held: AffinityFacts, whole: Whole) -> u32 {
        let index = match self.source {
            AffinitySource::Predictor => 0,
            AffinitySource::Index | AffinitySource::Merged => held.overlap_blocks,
        };
        let predicted = match self.source {
            AffinitySource::Index => 0,
            AffinitySource::Predictor | AffinitySource::Merged
                if whole.clears(held.predicted_blocks, self.threshold) =>
            {
                held.predicted_blocks
            }
            AffinitySource::Predictor | AffinitySource::Merged => 0,
        };
        index.max(predicted)
    }

    /// The stage's scores over what its two passes read of the kept rows.
    /// `feed` says whether anybody holds anything and carries the drawn
    /// rows' mean where the engine set one. `reads` hands each pass a
    /// reader of its own: a column's is a read by index, and the rows a
    /// host named are walked beside the kept rows. One body a feed, the
    /// feed chosen where the body is entered. The rows of a draw are
    /// scored by [`Self::rank_drawn`]. The count of the rows written the
    /// ceiling comes back beside the band: the entry that says the stage's
    /// word reads it ([`Scorer::score_front`]), and an entry that does not
    /// carries no count.
    #[inline(always)]
    fn rank<R: FnMut(u32) -> AffinityFacts>(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        feed: Feed<'_>,
        reads: impl Fn() -> R,
    ) -> (TieBand, u32) {
        let whole = self.gates.whole(facts);
        let holds = !feed.is_empty();
        // Nobody holding anything, the deepest match is zero without a
        // pass; any other feed answers for every kept row.
        let best = if holds {
            let mut entry = reads();
            kept.iter()
                .map(|&i| self.matched(entry(i), whole))
                .max()
                .unwrap_or(0)
        } else {
            0
        };
        let avg = spill_mean(feed.drawn_mean(), facts, candidates, kept);
        // A decode worker prefills nothing again: its held prefix is not
        // priced.
        let held = if self.priced(facts) {
            self.judge(best, avg);
            self.gates.tokens_of(best, facts)
        } else {
            0.0
        };
        let mut entry = reads();
        let mut ceilings = 0u32;
        with_counts!(facts, |counts| {
            for (o, &i) in out.iter_mut().zip(kept) {
                let c = &candidates[i as usize];
                let load = counts(c) as f32;
                // A match above zero was read from a feed that holds entries.
                *o = if best > 0
                    && self.matched(entry(i), whole) == best
                    && !self.gates.over(load, avg, held)
                {
                    // Counted where it is written: a row under the ceiling
                    // adds nothing to the pass.
                    ceilings += 1;
                    1.0
                } else if !self.gates.over(load, avg, 0.0) {
                    0.5
                } else {
                    0.0
                };
            }
        });
        (TieBand(0.0), ceilings)
    }

    /// The stage's scores behind a draw, as [`Self::rank`] says them over
    /// the rows a host named. The rows come in the draw's order, and no
    /// test of a row is one a processor predicts: each row's entry is
    /// found once, by a search (its match rides the row's score slot, bit
    /// for bit, to the pass that scores), the spill gate's bounds are
    /// taken once and both tested, and the score is chosen by a table. A
    /// pool read whole keeps [`Self::rank`]: its rows come in the host's
    /// order decision after decision, its tests are predicted, and this
    /// body costs it more instructions a row.
    fn rank_drawn(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        named: AffinityRows<'_>,
    ) -> TieBand {
        let whole = self.gates.whole(facts);
        // Nobody holding anything, the deepest match is zero without a
        // pass.
        let mut best = 0;
        if !named.holders.is_empty() {
            for (o, &i) in out.iter_mut().zip(kept) {
                let matched = self.matched(named.entry(i), whole);
                *o = f32::from_bits(matched);
                best = best.max(matched);
            }
        }
        let avg = spill_mean(named.mean_in_flight, facts, candidates, kept);
        let held = if self.priced(facts) {
            self.judge(best, avg);
            self.gates.tokens_of(best, facts)
        } else {
            0.0
        };
        let (holder, anyone) = (self.gates.spill(avg, held), self.gates.spill(avg, 0.0));
        with_counts!(facts, |counts| {
            for (o, &i) in out.iter_mut().zip(kept) {
                let load = counts(&candidates[i as usize]) as f32;
                // A match above zero was read into this slot.
                let deepest = (best > 0) & (o.to_bits() == best);
                let own = usize::from(deepest & !holder.over(load));
                *o = [0.0, 0.5, 1.0, 1.0][own << 1 | usize::from(!anyone.over(load))];
            }
        });
        TieBand(0.0)
    }

    /// The stage's scores over the rows a host named: a holder reads its
    /// depth. Each pass walks the holders beside its kept rows where the
    /// pool is read whole; behind a draw each row is searched once
    /// ([`Self::rank_drawn`]). Out of line: see [`Scorer::score`] of this
    /// stage.
    #[inline(never)]
    fn rank_named(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        named: AffinityRows<'_>,
    ) -> TieBand {
        if named.drawn() {
            // Behind a draw the rows stand in the draw's order: each is
            // read by a search, once, and no cursor is kept.
            return self.rank_drawn(facts, candidates, kept, out, named);
        }
        self.rank(facts, candidates, kept, out, Feed::Named(named), || {
            let mut walk = named.walk();
            move |i: u32| walk.entry(i)
        })
        .0
    }
}

impl Lifecycle for PrefixAffinity {}

impl Scorer for PrefixAffinity {
    fn name(&self) -> &'static str {
        "prefix-affinity"
    }

    // Scores by the affinity column. Its spill gate reads the mean count and places elsewhere.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// Row by row, unless the fleet's KV gate is configured: that gate
    /// reads every row.
    fn row_wise(&self) -> bool {
        !self.gates.kv_configured()
    }

    /// A holder of the deepest match, a row under the gate, a row over it.
    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Levels(0.5)
    }

    fn states_overlap(&self) -> bool {
        true
    }

    /// The row's match by the stage's source, of the request's blocks, and
    /// whether the held prefix's price kept the row in front: it is a
    /// deepest holder of the pass that decided, and at the mean that pass
    /// judged by it stands over the swing's margin and under the priced
    /// one. The row's count is the decision's own: a lent table answers a
    /// second read with the first. Under the fleet's KV gate the stage
    /// ordered nothing: the row's match is said all the same, and no
    /// margin.
    ///
    /// A request of more than 65,535 blocks halves both counts until its
    /// own fits, so the share they say stays true.
    fn said(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], row: u32) -> Overlap {
        let request_blocks = self.gates.request_blocks(facts);
        let held = facts.feed().entry_or_default(row);
        let blocks = self.matched(held, self.gates.whole(facts));
        let indexed = self.source != AffinitySource::Predictor && held.overlap_blocks == blocks;
        // The pass's word is taken whichever row is asked about: it is
        // the word of one decision.
        let judged = take_prefix_pass(self.id);
        let margin_held = blocks > 0
            && judged.is_some_and(|(best, mean)| {
                best == blocks
                    && candidates.get(row as usize).is_some_and(|candidate| {
                        let load = facts.in_flight(candidate) as f32;
                        let price = self.gates.tokens_of(blocks, facts);
                        self.gates.over(load, mean, 0.0) && !self.gates.over(load, mean, price)
                    })
            });
        // Both counts by one shift, and the source, where every statement
        // of an overlap ends.
        Overlap::of(blocks, indexed, request_blocks, margin_held)
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        self.score_front_with(facts, candidates, kept, asked, out, |c| c.load.as_ref())
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.score_with(facts, candidates, kept, out, |c| c.load.as_ref())
    }

    fn needs(&self) -> Needs {
        let placements = self.source != AffinitySource::Index;
        Needs {
            // Every source reads the column: the index's overlap and the
            // predictor's placement are fields of a candidate's entry. What
            // feeds the entry differs by source and is declared beside it.
            affinity: true,
            token_hashes: true,
            events: self.source != AffinitySource::Predictor,
            // The record of sent prefixes holds text windows beside token
            // blocks, and the events' index holds token blocks alone: a
            // host hashes its text for a source that reads the record.
            sent_prefixes: placements,
            text_hashes: placements,
            // A sent prefix's share is taken of the prompt's size: a host
            // that can send a request on unread reads it under this stage.
            prompt_size: true,
            loads: self.gates.kv_configured(),
            ..Needs::default()
        }
    }
}

impl ReadsLoad for PrefixAffinity {
    /// The pass counts the rows it writes the ceiling. Under the fleet's
    /// KV gate every row stands alike. The rows a host named say nothing.
    fn score_front_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &[u32],
        asked: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> (TieBand, ScoreFront) {
        if facts.named_alone().is_some() {
            return (
                self.score_with(facts, candidates, asked, out, load),
                ScoreFront::Unsaid,
            );
        }
        if self.gates.kv_imbalanced(candidates, asked, load) {
            self.unjudge();
            return (neutral(out), ScoreFront::Alike);
        }
        let column = facts.affinity;
        let (band, ceilings) =
            self.rank(facts, candidates, asked, out, Feed::Column(column), || {
                |i: u32| column[i as usize]
            });
        // The values are the ceiling, a half and nothing: every row not
        // counted stands a level or more under the rows that are.
        let said = if ceilings as usize == out.len() {
            ScoreFront::Alike
        } else {
            ScoreFront::At(ceilings)
        };
        (band, said)
    }

    // The front's named path calls this whole scorer. Keep it out of
    // the pass over a column.
    #[inline(never)]
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        if self.gates.kv_imbalanced(candidates, kept, load) {
            self.unjudge();
            return neutral(out);
        }
        // What the rows hold is taken once a pass: a read by index tests
        // the feed and its bound for every row. The rows a host named
        // answer out of line, so a pass over a column keeps the code it
        // had: one entry a candidate, and a kept row reads its own.
        if let Some(named) = facts.named_alone() {
            return self.rank_named(facts, candidates, kept, out, named);
        }
        let column = facts.affinity;
        self.rank(facts, candidates, kept, out, Feed::Column(column), || {
            |i: u32| column[i as usize]
        })
        .0
    }
}

// ---------------------------------------------------------------- kv pressure

/// The fleet KV gate: its parameters configure the gate every affinity stage
/// of the profile reads (affinity yields to load past the usage spread or the
/// ceiling), and the stage itself orders nothing.
#[derive(Debug)]
pub struct KvPressure;

impl Lifecycle for KvPressure {}

impl Scorer for KvPressure {
    fn name(&self) -> &'static str {
        "kv-pressure"
    }

    // Scores by the pressure its feed reports.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The stage orders nothing: the gate it configures is read by the
    /// profile's affinity stages, which answer for it.
    fn row_wise(&self) -> bool {
        true
    }

    /// Every row reads alike: the stage configures the gate and orders
    /// nothing.
    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Alike
    }

    fn score(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &[u32], out: &mut [f32]) -> TieBand {
        neutral(out)
    }

    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            ..Needs::default()
        }
    }
}

// ---------------------------------------------------------------- warm-up slice

/// The warm-up slice. One miss in `1 / share` goes to the least-loaded
/// warming worker (one with a thin index against the fleet's level, or a
/// young one short of its first blocks), so a returned, new or emptied worker
/// builds a cache instead of idling behind the fleet's affinity; one hit in
/// `divert_every` goes to the least-loaded thin worker although another
/// holds its prefix (shallow hits first, a deep hit only away from a holder
/// with more in flight), each thin worker at most once per window while it
/// has anything in flight. When the stage takes a request its worker scores
/// `1.0` and every other `0.0`; otherwise every row scores alike. A fleet
/// with no warming worker returns before any counter moves.
#[derive(Debug)]
pub struct WarmupSlice {
    secs_micros: u64,
    share: f32,
    blocks: u32,
    thin_ratio: f32,
    divert_every: u64,
    miss_share: f32,
    gates: Gates,
    /// Misses the slice saw; every `period`th goes to a warming worker.
    misses: AtomicU64,
    /// Hits since the last diversion, and whether a shallow one was among
    /// them.
    hits: AtomicU64,
    shallow_seen: AtomicBool,
    /// The decision clock before which a thin worker with requests in flight
    /// takes no further diverted hit, by worker id.
    divert_until: Mutex<FxHashMap<WorkerId, u64>>,
    /// The fleet's index sizes, for the level: one buffer, reused.
    sizes: Mutex<Vec<u32>>,
}

impl WarmupSlice {
    /// The stage from its parameters and its profile's gates.
    pub fn new(params: &WarmupSliceParams, gates: Gates) -> Self {
        Self {
            secs_micros: params
                .secs
                .unwrap_or(DEFAULT_WARMUP_SECS)
                .saturating_mul(1_000_000),
            share: params.share.unwrap_or(DEFAULT_WARMUP_SHARE),
            blocks: u32::try_from(params.blocks.unwrap_or(DEFAULT_WARMUP_BLOCKS))
                .unwrap_or(u32::MAX),
            thin_ratio: params.thin_ratio.unwrap_or(DEFAULT_WARMUP_THIN_RATIO),
            divert_every: params.divert_every.unwrap_or(DEFAULT_WARMUP_DIVERT_EVERY),
            miss_share: params.miss_share.unwrap_or(DEFAULT_THRESHOLD),
            gates,
            misses: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            shallow_seen: AtomicBool::new(false),
            divert_until: Mutex::new(FxHashMap::default()),
            sizes: Mutex::new(Vec::new()),
        }
    }

    /// The stage's ruling of who warms, as a host enters it.
    #[inline]
    fn rule(&self) -> WarmUpRule {
        WarmUpRule {
            window_micros: self.secs_micros,
            blocks: self.blocks,
            thin_ratio: self.thin_ratio,
        }
    }

    /// Every how many misses one goes to a warming worker.
    fn period(&self) -> u64 {
        if self.share <= 0.0 {
            return u64::MAX;
        }
        ((1.0 / f64::from(self.share)).round() as u64).max(1)
    }

    /// Whether an index of `indexed` blocks is thin against the fleet's
    /// `level`: empty, or below the ratio of the level, once the fleet holds
    /// a cache worth catching up to.
    fn is_thin(&self, indexed: u32, level: u32) -> bool {
        self.rule().is_thin(indexed, level)
    }

    /// Whether a worker is young: admitted within the window and short of
    /// its first blocks.
    fn is_young(&self, held: AffinityFacts) -> bool {
        let age = u64::from(held.admitted_age_ms).saturating_mul(1_000);
        self.rule().is_young(age, held.indexed_growth)
    }

    /// The fleet's index level: the upper median of the index sizes in the
    /// decision's column, so one emptied worker does not drag it down.
    fn fleet_level(&self, column: &[AffinityFacts]) -> u32 {
        let mut sizes = self.sizes.lock();
        sizes.clear();
        sizes.extend(column.iter().map(|held| held.indexed_blocks));
        WarmUpRule::level(&mut sizes)
    }

    /// The stage over the rows a host named: it ruled the warming rows
    /// where it wrote, and none named slices nothing. Out of line, so a
    /// pass over a column keeps the code it had.
    #[inline(never)]
    fn score_named(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        named: AffinityRows<'_>,
    ) -> TieBand {
        if named.warming.is_empty() {
            return neutral(out);
        }
        let warm = Warm::Named {
            rows: named.warming,
            thin: named.thin,
        };
        // Behind a draw the pass reads the rows named by a search: one
        // body a read, the read chosen where the body is entered.
        if named.drawn() {
            return self.turn::<true>(facts, candidates, kept, out, Feed::Named(named), warm);
        }
        self.turn::<false>(facts, candidates, kept, out, Feed::Named(named), warm)
    }

    /// One pass where somebody warms: the request is a miss the slice may
    /// take or a hit the divert may take, and the row that takes it scores
    /// alone. `DRAWN` is a pass behind a draw: it reads the rows named by a
    /// search and keeps no cursor.
    #[inline(always)]
    fn turn<const DRAWN: bool>(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        feed: Feed<'_>,
        warm: Warm<'_>,
    ) -> TieBand {
        let request_blocks = self.gates.request_blocks(facts);
        let mut reads = feed.reads();
        let best = kept
            .iter()
            .map(|&i| feed.read::<DRAWN>(&mut reads, i).overlap_blocks)
            .max()
            .unwrap_or(0);
        let thin_overlap = best as f32 / request_blocks as f32 <= self.miss_share
            || (best <= WARMUP_MISS_BLOCKS && best * 2 < request_blocks);
        let mut pass = Pass {
            facts,
            candidates,
            kept,
            feed,
            warm,
            best,
            request_blocks,
            draw: Draw::new(facts.seed ^ 0x77a7_2d5e_6f1c_3b09),
        };
        let slot = if thin_overlap {
            self.slice::<DRAWN>(&mut pass)
        } else if warm.thin() {
            self.divert::<DRAWN>(&mut pass, facts.now_micros)
        } else {
            None
        };
        let Some(slot) = slot else {
            return neutral(out);
        };
        out.fill(0.0);
        out[slot] = 1.0;
        TieBand(0.0)
    }

    /// How a pass tells a warming row over a host's column: `None` when
    /// nobody warms. It takes one pass over the column. The fleet's median
    /// is read only where its smallest index is thin against its largest.
    #[inline(always)]
    fn ruled(&self, column: &[AffinityFacts]) -> Option<Warm<'static>> {
        let mut smallest = u32::MAX;
        let mut largest = 0;
        let mut young = 0usize;
        for &held in column {
            smallest = smallest.min(held.indexed_blocks);
            largest = largest.max(held.indexed_blocks);
            young += usize::from(self.is_young(held));
        }
        if largest == 0 && young == 0 {
            return None;
        }
        // Even the largest index is too low a level to make anyone thin
        // if the smallest index passes at that level.
        let (level, thin_any) = if self.is_thin(smallest, largest) {
            let level = self.fleet_level(column);
            (level, self.is_thin(smallest, level))
        } else {
            (0, false)
        };
        if !thin_any && (young == 0 || young == column.len()) {
            return None;
        }
        Some(Warm::Ruled {
            level,
            thin_any,
            young_all: young == column.len(),
        })
    }

    /// How one pass tells its warming rows, asked in the pass's own order.
    /// A thin worker is one whatever the fleet's age; workers warming only
    /// because they are young are ones only when the fleet is not all
    /// young. A host that named the rows ruled the same where it derived
    /// them: the pass walks its list beside the kept rows of a pool read
    /// whole, and searches it a row behind a draw (`DRAWN`).
    fn warming<'a, const DRAWN: bool>(
        &'a self,
        warm: Warm<'a>,
        feed: Feed<'a>,
    ) -> impl FnMut(u32) -> bool + 'a {
        let mut reads = feed.reads();
        let mut named = Walk::new(match warm {
            Warm::Named { rows, .. } => rows,
            Warm::Ruled { .. } => &[],
        });
        move |i| match warm {
            Warm::Ruled {
                level,
                thin_any,
                young_all,
            } => {
                let held = reads.entry(i);
                if thin_any {
                    self.is_thin(held.indexed_blocks, level)
                } else {
                    !young_all && self.is_young(held)
                }
            }
            Warm::Named { rows, .. } if DRAWN => rows.binary_search(&i).is_ok(),
            Warm::Named { .. } => named.find(i, |&row| row).is_some(),
        }
    }

    /// The slot of the kept row the slice sends this miss to, if it is due.
    fn slice<const DRAWN: bool>(&self, pass: &mut Pass<'_>) -> Option<usize> {
        let avg = spill_mean(
            pass.feed.drawn_mean(),
            pass.facts,
            pass.candidates,
            pass.kept,
        );
        let mut warming = self.warming::<DRAWN>(pass.warm, pass.feed);
        let slot =
            pass.least_loaded(|i, _, load| warming(i) && !self.gates.over(load as f32, avg, 0.0))?;
        let seen = self.misses.fetch_add(1, Ordering::Relaxed);
        seen.is_multiple_of(self.period()).then_some(slot)
    }

    /// The slot of the kept row a hit is diverted to, if one is due.
    fn divert<const DRAWN: bool>(&self, pass: &mut Pass<'_>, now: u64) -> Option<usize> {
        if self.divert_every == 0 {
            return None;
        }
        let Pass {
            facts,
            candidates,
            kept,
            feed,
            warm,
            best,
            request_blocks,
            ..
        } = *pass;
        let shallow = best as f32 / request_blocks as f32 <= DIVERT_SHALLOW_SHARE;
        if shallow {
            self.shallow_seen.store(true, Ordering::Relaxed);
        }
        let hits = self.hits.fetch_add(1, Ordering::Relaxed).wrapping_add(1);
        let shallow_seen = self.shallow_seen.load(Ordering::Relaxed);
        let due = (hits >= self.divert_every && (shallow || !shallow_seen))
            || hits >= self.divert_every.saturating_mul(2);
        if !due {
            return None;
        }
        // What a deep hit's thin worker is measured against: the least
        // loaded of the request's deepest holders. A thin worker among them
        // recomputes nothing and is exempt.
        let holder_load = if shallow {
            None
        } else {
            let mut reads = feed.reads();
            kept.iter()
                .filter(|&&i| feed.read::<DRAWN>(&mut reads, i).overlap_blocks == best)
                .map(|&i| facts.in_flight(&candidates[i as usize]))
                .min()
        };
        // The eligibility read and the window's reservation on one lock
        // turn: two decisions due together cannot both divert to the same
        // thin worker inside one window. A hit is diverted only where the
        // warming rows are the thin, so a warming row here is a thin one.
        let slot = {
            let mut until = self.divert_until.lock();
            let (mut warming, mut held) = (self.warming::<DRAWN>(warm, feed), feed.reads());
            let slot = pass.least_loaded(|i, c, load| {
                warming(i)
                    && !(load > 0 && until.get(&c.id).is_some_and(|&t| now < t))
                    && !(holder_load.is_some_and(|h| load >= h)
                        && feed.read::<DRAWN>(&mut held, i).overlap_blocks != best)
            })?;
            until.insert(
                candidates[kept[slot] as usize].id,
                now + DIVERT_WINDOW_MICROS,
            );
            slot
        };
        self.hits.store(0, Ordering::Relaxed);
        self.shallow_seen.store(false, Ordering::Relaxed);
        Some(slot)
    }
}

impl Lifecycle for WarmupSlice {
    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            Membership::Removed(id)
            | Membership::Replaced { old: id, .. }
            | Membership::Resized(id) => {
                self.divert_until.lock().remove(id);
            }
            Membership::Snapshot(ids) => {
                self.divert_until
                    .lock()
                    .retain(|worker, _| ids.contains(worker));
            }
            Membership::Added(_) | Membership::Relabelled(_) | Membership::Left { .. } => {}
        }
    }
}

impl Scorer for WarmupSlice {
    fn name(&self) -> &'static str {
        "warmup-slice"
    }

    // Scores by the warm-up state. Its divert reads a holder's count and places elsewhere.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// Row by row, unless the fleet's KV gate is configured: that gate
    /// reads every row. What the stage reads of the pool (its level, who
    /// warms) comes from the host's feed, never from the rows handed.
    fn row_wise(&self) -> bool {
        !self.gates.kv_configured()
    }

    /// The row that takes the request scores alone, or every row alike.
    fn spread(&self) -> ScoreSpread {
        ScoreSpread::Levels(1.0)
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        self.score_front_with(facts, candidates, kept, asked, out, |c| c.load.as_ref())
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        self.score_with(facts, candidates, kept, out, |c| c.load.as_ref())
    }

    fn needs(&self) -> Needs {
        Needs {
            // The admitted age, the index's growth and size and the overlap
            // are all read from a candidate's entry of the column.
            affinity: true,
            events: true,
            loads: true,
            // A miss is an overlap that is thin against the prompt's size.
            prompt_size: true,
            ..Needs::default()
        }
    }

    fn warm_up_rule(&self) -> Option<WarmUpRule> {
        Some(self.rule())
    }
}

impl ReadsLoad for WarmupSlice {
    /// A slice either scores every row alike or gives one row the ceiling.
    /// The first two scores tell those answers apart. A lone row stands at
    /// the ceiling either way. The rows a host named say nothing.
    fn score_front_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _: &[u32],
        asked: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> (TieBand, ScoreFront) {
        let band = self.score_with(facts, candidates, asked, out, load);
        if facts.named_alone().is_some() {
            return (band, ScoreFront::Unsaid);
        }
        let alike = match *out {
            [first, second, ..] => first >= 1.0 && second >= 1.0,
            _ => true,
        };
        (
            band,
            if alike {
                ScoreFront::Alike
            } else {
                ScoreFront::At(1)
            },
        )
    }

    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        if self.share <= 0.0 || self.gates.kv_imbalanced(candidates, kept, load) {
            return neutral(out);
        }
        // The rows a host named answer out of line, so a pass over a column
        // keeps the code it had.
        if let Some(named) = facts.named_alone() {
            return self.score_named(facts, candidates, kept, out, named);
        }
        // A decision fed no column has nothing to slice: every row reads as
        // just admitted with nothing indexed, a young fleet, and a young
        // fleet slices nothing.
        if facts.affinity.is_empty() {
            return neutral(out);
        }
        // The column is taken once: one entry a candidate, walked by the
        // passes over the pool and indexed by the reads of a kept row. One
        // that ends before the candidates is a host's mistake and fails
        // here, as an index past its end does.
        let column = &facts.affinity[..candidates.len()];
        let Some(warm) = self.ruled(column) else {
            return neutral(out);
        };
        self.turn::<false>(facts, candidates, kept, out, Feed::Column(column), warm)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        document::spec::{PolicySpec, SpillParams},
        engine::{
            decision::OverlapSource,
            framework::{Engine, Scratch},
            prefix_pass::forget_prefix_pass,
            view::{AffinityHolder, Constraints, Health, LoadRow, PoolFacts},
        },
        tracking::in_flight::tests::Lent,
    };

    /// A row and its entry of the affinity column, as a case writes a pool.
    type Held = (Candidate, AffinityFacts);

    fn candidate(position: u32, in_flight: u32, overlap_blocks: u32) -> Held {
        (
            Candidate {
                position,
                id: WorkerId(u64::from(position) + 1),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight,
                ..Candidate::default()
            },
            AffinityFacts {
                overlap_blocks,
                admitted_age_ms: u32::MAX,
                ..AffinityFacts::default()
            },
        )
    }

    fn rows(pool: &[Held]) -> Vec<Candidate> {
        pool.iter().map(|(row, _)| *row).collect()
    }

    fn column(pool: &[Held]) -> Vec<AffinityFacts> {
        pool.iter().map(|(_, held)| *held).collect()
    }

    fn kept(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    fn facts(blocks: usize) -> RequestFacts<'static> {
        static HASHES: [u64; 64] = [0; 64];
        RequestFacts {
            block_hashes: &HASHES[..blocks],
            prompt_tokens: 16 * blocks as u32,
            seed: 7,
            ..RequestFacts::default()
        }
    }

    fn affinity(spill: Option<SpillParams>) -> PrefixAffinity {
        let params = PrefixAffinityParams {
            spill,
            ..PrefixAffinityParams::default()
        };
        PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)))
    }

    /// The scores of `pool`'s rows, the facts carrying the pool's column.
    fn scores(scorer: &dyn Scorer, facts: &RequestFacts<'_>, pool: &[Held]) -> Vec<f32> {
        let column = column(pool);
        scores_beside(scorer, facts, &rows(pool), &column)
    }

    /// The scores of `rows` beside a column of the case's own length.
    fn scores_beside(
        scorer: &dyn Scorer,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        column: &[AffinityFacts],
    ) -> Vec<f32> {
        // The counts a case wrote into its rows stand in a table lent
        // beside them, each row reading zero: every case that scores
        // through here holds its stage to the one reader of a row's count.
        let mut rows = rows.to_vec();
        let lent = Lent::take(&mut rows);
        let facts = lent.beside(&RequestFacts {
            affinity: column,
            ..*facts
        });
        let kept = kept(rows.len() as u32);
        let mut out = vec![0.0; rows.len()];
        scorer.score(&facts, &rows, &kept, &mut out);
        out
    }

    #[test]
    fn the_deepest_holders_rank_first_and_a_miss_leaves_every_row_alike() {
        let pool = [candidate(0, 1, 2), candidate(1, 1, 8), candidate(2, 1, 8)];
        assert_eq!(
            scores(&affinity(None), &facts(16), &pool),
            vec![0.5, 1.0, 1.0]
        );
        let miss = [candidate(0, 1, 0), candidate(1, 1, 0)];
        assert_eq!(scores(&affinity(None), &facts(16), &miss), vec![0.5, 0.5]);
    }

    #[test]
    fn a_holder_over_the_spill_gate_yields_to_the_rows_under_it() {
        // Mean 10 in flight; the holder at 40 is over 1.5 times the mean and
        // over the mean by the margin (min(64, max(ceil(2 * sqrt(10)), 4))
        // = 7), so it spills; the other rows are under the gate.
        let pool = [candidate(0, 40, 8), candidate(1, 0, 0), candidate(2, 0, 2)];
        let gated = affinity(Some(SpillParams {
            abs: Some(64),
            rel: Some(1.5),
            ..SpillParams::default()
        }));
        let mean = mean_in_flight(&RequestFacts::default(), &rows(&pool), &kept(3));
        assert!((mean - 40.0 / 3.0).abs() < 1e-6);
        assert_eq!(scores(&gated, &facts(16), &pool), vec![0.0, 0.5, 0.5]);
        // Under a lifted relative part the holder keeps the request.
        let lifted = affinity(Some(SpillParams {
            rel: Some(4.0),
            ..SpillParams::default()
        }));
        assert_eq!(scores(&lifted, &facts(16), &pool), vec![1.0, 0.5, 0.5]);
    }

    #[test]
    fn a_priced_prefix_keeps_its_holder_through_a_skew_the_swing_would_spill() {
        // Mean 10: the swing margin is 7 and the holder at 20 is over it. A
        // prefix of 8 blocks of 16 tokens priced at 8 tokens a request is
        // worth 16 requests, so the margin is 16 and the holder stays.
        let pool = [candidate(0, 20, 8), candidate(1, 0, 0)];
        let unpriced = affinity(Some(SpillParams {
            rel: Some(1.1),
            ..SpillParams::default()
        }));
        assert_eq!(scores(&unpriced, &facts(16), &pool), vec![0.0, 0.5]);
        let priced = affinity(Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens: Some(8),
            ..SpillParams::default()
        }));
        assert_eq!(scores(&priced, &facts(16), &pool), vec![1.0, 0.5]);
        // The price reads the block size the request was counted with: at
        // one token a block the eight blocks are one request, the margin
        // falls back to the swing and the holder spills; at 128 a block
        // they are 128 requests, capped at the absolute 64, and it stays.
        let narrow = RequestFacts {
            block_size: 1,
            ..facts(16)
        };
        assert_eq!(scores(&priced, &narrow, &pool), vec![0.0, 0.5]);
        let wide = RequestFacts {
            block_size: 128,
            ..facts(16)
        };
        assert_eq!(scores(&priced, &wide, &pool), vec![1.0, 0.5]);
        // A decode decision prices nothing.
        let decode = RequestFacts {
            phase: Phase::Decode,
            ..facts(16)
        };
        assert_eq!(scores(&priced, &decode, &pool), vec![0.0, 0.5]);
    }

    #[test]
    fn a_predicted_placement_counts_only_above_the_threshold_and_the_index_at_any_depth() {
        let params = PrefixAffinityParams {
            threshold: Some(0.5),
            source: Some(AffinitySource::Merged),
            ..PrefixAffinityParams::default()
        };
        let merged = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
        let predicted = |(row, held): Held, predicted_blocks: u32| {
            (
                row,
                AffinityFacts {
                    predicted_blocks,
                    ..held
                },
            )
        };
        let shallow = predicted(candidate(0, 0, 0), 4);
        let deep = predicted(candidate(1, 0, 0), 9);
        let indexed = candidate(2, 0, 1);
        // 4 of 16 is under the threshold, 9 of 16 over it; the index's one
        // block is a match whatever the threshold, and the deepest wins.
        assert_eq!(
            scores(&merged, &facts(16), &[shallow, deep, indexed]),
            vec![0.5, 1.0, 0.5]
        );
        assert_eq!(
            scores(&merged, &facts(16), &[shallow, indexed]),
            vec![0.5, 1.0]
        );
        // Without hashes the request is sized by its tokens.
        let by_tokens = RequestFacts {
            block_hashes: &[],
            prompt_tokens: 256,
            ..RequestFacts::default()
        };
        assert_eq!(
            scores(&merged, &by_tokens, &[shallow, deep]),
            vec![0.5, 1.0]
        );
    }

    /// A sent prefix's share is taken of the whole prompt, as the classic
    /// policy takes it: its matched tokens over the prompt's tokens, above
    /// the threshold (`model_gateway/src/policies/cache_aware.rs:3561-3569`).
    /// A prompt that ends inside a block is larger than its whole blocks, so
    /// a share of those would let a prefix under the threshold count. The
    /// share is one division of the two counts: at the threshold itself it
    /// does not clear it.
    #[test]
    fn a_sent_prefix_counts_by_its_share_of_the_whole_prompt() {
        static HASHES: [u64; 64] = [0; 64];
        // Whether `held` blocks of `block` tokens sent to a worker count
        // for a prompt of `tokens` tokens, hashed by its whole blocks.
        let counts = |threshold: Option<f32>, block: u32, held: u32, tokens: u32| {
            let params = PrefixAffinityParams {
                threshold,
                source: Some(AffinitySource::Predictor),
                ..PrefixAffinityParams::default()
            };
            let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(block)));
            let (row, nothing) = candidate(0, 0, 0);
            let sent = AffinityFacts {
                predicted_blocks: held,
                ..nothing
            };
            let facts = RequestFacts {
                block_hashes: &HASHES[..(tokens / block) as usize],
                prompt_tokens: tokens,
                seed: 7,
                ..RequestFacts::default()
            };
            let scored = scores(&stage, &facts, &[(row, sent), candidate(1, 0, 0)]);
            assert!(scored == [1.0, 0.5] || scored == [0.5, 0.5], "{scored:?}");
            scored[0] == 1.0
        };
        // One block of 16 tokens at the default share of 0.3: it is 0.302 of
        // 53 tokens and 0.296 of 54, and a third of either prompt's three
        // whole blocks.
        let counted = [48, 53, 54, 63, 64].map(|tokens| counts(None, 16, 1, tokens));
        assert_eq!(counted, [true, true, false, false, false]);
        // Four blocks: 0.3005 of 213 tokens, 0.2991 of 214.
        assert!(counts(None, 16, 4, 213));
        assert!(!counts(None, 16, 4, 214));
        // 72 tokens of 160 are 0.45 to the bit, and do not clear 0.45. Three
        // blocks over 160 / 24 blocks would read one bit above it.
        assert!(!counts(Some(0.45), 24, 3, 160));
        assert!(counts(Some(0.45), 24, 3, 159));
        // The whole keeps what the request's block count guards: a size
        // stated under the hashes the host sent reads as the hashes, a
        // prompt of no size as one block, and the text kind's as its bytes
        // beside a window of bytes. A request of text states its length and
        // no count of tokens: 100 two-byte characters are 200 bytes, and
        // the share is taken of the bytes.
        let whole = |facts: RequestFacts<'_>| {
            let whole = Gates::of(None, None, Some(16)).whole(&facts);
            (whole.block, whole.size)
        };
        let hashed = RequestFacts {
            block_hashes: &HASHES[..3],
            ..RequestFacts::default()
        };
        let sized = |prompt_tokens| RequestFacts {
            prompt_tokens,
            ..hashed
        };
        assert_eq!(whole(sized(54)), (16, 54));
        assert_eq!(whole(sized(20)), (16, 48));
        assert_eq!(whole(RequestFacts::default()), (16, 16));
        let text = RequestFacts {
            hash_kind: HashKind::Text,
            block_size: 64,
            prompt_chars: 100,
            prompt_bytes: 200,
            ..hashed
        };
        assert_eq!(whole(text), (64, 200));
    }

    /// The engine of the built-in `cache_aware` document.
    fn built_in() -> Engine {
        PolicySpec::preset("cache_aware")
            .expect("a preset")
            .validate()
            .expect("the preset validates")
    }

    /// The built-in document's decision over `pool`, a pool of more than
    /// twice its sample of sixteen rows, for a prompt of `tokens` tokens in
    /// three whole blocks whose first block was sent to row 7: the row
    /// picked. The host names row 7, and the decision runs behind a draw.
    fn decided_past_the_sample(engine: &Engine, pool: &[Candidate], tokens: u32, seed: u64) -> u32 {
        static HASHES: [u64; 3] = [0; 3];
        let sent = [AffinityHolder {
            row: 7,
            overlap_blocks: 0,
            predicted_blocks: 1,
        }];
        let facts = RequestFacts {
            model: "m",
            block_hashes: &HASHES,
            prompt_tokens: tokens,
            seed: Draw::new(seed).next_u64(),
            named: Some(named(&sent, &[], false)),
            ..RequestFacts::default()
        };
        let decision = engine.decide(
            &facts,
            &PoolFacts::default(),
            pool,
            &Constraints::default(),
            &mut Scratch::new(),
        );
        assert!(decision.trace().drew_first(), "seed {seed}");
        let pick = decision.pick().expect("a pick");
        pick.targets.first().expect("a target").position
    }

    /// Behind a draw the share is the same one. A pool of more than twice
    /// the built-in document's sample is decided over the rows drawn and the
    /// rows its host named, and 16 tokens sent of a prompt of 54 hold
    /// nothing there: they are under 0.3 of the prompt, and a third of its
    /// three whole blocks. The classic policy divides the same two counts,
    /// and under its threshold it selects with no holder named
    /// (`model_gateway/src/policies/cache_aware.rs:3561-3569`, `:3579-3588`).
    #[test]
    fn behind_a_draw_a_sent_prefix_counts_by_its_share_of_the_whole_prompt() {
        let engine = built_in();
        // Sixty-four rows. Row 7 was sent the prefix and is the one busy
        // row: it takes a request it holds, and an idle row takes any other.
        let pool: Vec<Candidate> = (0..64)
            .map(|row| candidate(row, if row == 7 { 2 } else { 0 }, 0).0)
            .collect();
        for seed in 0..32 {
            // 0.302 of 53 tokens, 0.296 of 54.
            let held = decided_past_the_sample(&engine, &pool, 53, seed);
            assert_eq!(held, 7, "53 tokens, seed {seed}");
            let missed = decided_past_the_sample(&engine, &pool, 54, seed);
            assert_ne!(missed, 7, "54 tokens, seed {seed}");
        }
    }

    fn reporting(position: u32, in_flight: u32, overlap_blocks: u32, active_kv: u16) -> Held {
        let (row, held) = candidate(position, in_flight, overlap_blocks);
        (
            Candidate {
                load: Some(LoadRow {
                    active_kv_permille: active_kv,
                    kv_usage_permille: active_kv,
                    live_rate: 1000,
                    ..LoadRow::default()
                }),
                ..row
            },
            held,
        )
    }

    #[test]
    fn under_the_fleet_kv_gate_affinity_yields_to_load() {
        let kv = KvPressureParams {
            spread: Some(0.3),
            ceiling: Some(0.9),
            ..KvPressureParams::default()
        };
        let params = PrefixAffinityParams::default();
        let gated = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
        let balanced = [reporting(0, 0, 8, 500), reporting(1, 0, 0, 450)];
        assert_eq!(scores(&gated, &facts(16), &balanced), vec![1.0, 0.5]);
        let spread = [reporting(0, 0, 8, 800), reporting(1, 0, 0, 300)];
        assert_eq!(scores(&gated, &facts(16), &spread), vec![1.0, 1.0]);
        let ceiling = [reporting(0, 0, 8, 950), reporting(1, 0, 0, 900)];
        assert_eq!(scores(&gated, &facts(16), &ceiling), vec![1.0, 1.0]);
        // The gate at its defaults never fires, whatever the usage.
        let off = affinity(None);
        assert_eq!(scores(&off, &facts(16), &ceiling), vec![1.0, 0.5]);
        assert!(KvPressure.needs().loads);
        assert_eq!(scores(&KvPressure, &facts(16), &ceiling), vec![1.0, 1.0]);
    }

    fn warmup(share: f32, divert_every: u64) -> WarmupSlice {
        let params = WarmupSliceParams {
            share: Some(share),
            divert_every: Some(divert_every),
            blocks: Some(8),
            ..WarmupSliceParams::default()
        };
        WarmupSlice::new(&params, Gates::of(None, None, Some(16)))
    }

    fn sized(position: u32, in_flight: u32, overlap_blocks: u32, indexed: u32) -> Held {
        let (row, held) = candidate(position, in_flight, overlap_blocks);
        (
            row,
            AffinityFacts {
                indexed_blocks: indexed,
                indexed_growth: indexed,
                ..held
            },
        )
    }

    #[test]
    fn one_thin_miss_in_a_period_goes_to_the_least_loaded_thin_worker() {
        let slice = warmup(0.25, 0);
        // Level 64 (the upper median of 64, 64, 0): the empty worker is thin;
        // nobody holds the request.
        let pool = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
        let taken: Vec<bool> = (0..8)
            .map(|_| scores(&slice, &facts(16), &pool) == vec![0.0, 0.0, 1.0])
            .collect();
        assert_eq!(
            taken,
            vec![true, false, false, false, true, false, false, false]
        );
        // A settled fleet moves no counter and scores alike.
        let settled = [sized(0, 3, 0, 64), sized(1, 2, 0, 64)];
        assert_eq!(scores(&slice, &facts(16), &settled), vec![1.0, 1.0]);
        assert_eq!(slice.misses.load(Ordering::Relaxed), 8);
    }

    /// Two thin workers part by the counts the request's facts read: beside
    /// a lent table that the rows' own fields contradict, a sliced miss
    /// goes to the thin worker the table says is the least loaded.
    #[test]
    fn a_thin_miss_goes_to_the_thin_worker_the_lent_counts_say_is_least_loaded() {
        let slice = warmup(0.25, 0);
        // Level 64 (the median of 64, 64, 64, 0, 0): rows 3 and 4 are thin,
        // and row 4 has the fewer requests by the counts lent.
        let pool = [
            sized(0, 3, 0, 64),
            sized(1, 2, 0, 64),
            sized(2, 4, 0, 64),
            sized(3, 5, 0, 0),
            sized(4, 1, 0, 0),
        ];
        let column = column(&pool);
        let mut rows = rows(&pool);
        let lent = Lent::take(&mut rows);
        // The rows' own fields say the contrary.
        rows[3].in_flight = 1;
        rows[4].in_flight = 5;
        let facts = lent.beside(&RequestFacts {
            affinity: &column,
            ..facts(16)
        });
        let kept = kept(rows.len() as u32);
        let mut out = vec![0.0; rows.len()];
        slice.score(&facts, &rows, &kept, &mut out);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// A few-block hit on a long request is a miss for the slice: at most
    /// four blocks held that cover less than half the request. A fifth
    /// block, or half the request, is a hit the slice leaves alone.
    #[test]
    fn a_few_block_hit_on_a_long_request_is_sliced_as_a_miss() {
        let params = WarmupSliceParams {
            share: Some(1.0),
            divert_every: Some(0),
            blocks: Some(8),
            miss_share: Some(0.0),
            ..WarmupSliceParams::default()
        };
        let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
        let held = |blocks: u32| {
            [
                sized(0, 3, blocks, 64),
                sized(1, 2, 0, 64),
                sized(2, 5, 0, 0),
            ]
        };
        // Three of sixteen blocks: a miss, the thin worker takes it.
        assert_eq!(scores(&slice, &facts(16), &held(3)), vec![0.0, 0.0, 1.0]);
        // Five blocks: over the few-block bound, a hit.
        assert_eq!(scores(&slice, &facts(16), &held(5)), vec![1.0, 1.0, 1.0]);
        // Four of eight blocks cover half the request: a hit; three of
        // eight do not: a miss.
        assert_eq!(scores(&slice, &facts(8), &held(4)), vec![1.0, 1.0, 1.0]);
        assert_eq!(scores(&slice, &facts(8), &held(3)), vec![0.0, 0.0, 1.0]);
    }

    /// The host caps the hash list it sends: a request past the cap keeps
    /// its token size, so a few-block match on it is not read as half of it.
    #[test]
    fn a_request_past_the_hash_cap_keeps_its_token_size() {
        let params = WarmupSliceParams {
            share: Some(1.0),
            divert_every: Some(0),
            blocks: Some(8),
            miss_share: Some(0.0),
            ..WarmupSliceParams::default()
        };
        let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
        let held = |blocks: u32| {
            [
                sized(0, 3, blocks, 64),
                sized(1, 2, 0, 64),
                sized(2, 5, 0, 0),
            ]
        };
        // Eight hashes of a sixteen-block request: four blocks are a
        // quarter of it, a miss; of the hashes alone they would read half.
        let mut capped = facts(8);
        capped.prompt_tokens = 16 * 16;
        assert_eq!(scores(&slice, &capped, &held(4)), vec![0.0, 0.0, 1.0]);
        assert_eq!(scores(&slice, &capped, &held(8)), vec![1.0, 1.0, 1.0]);
        // A text-kind request states its length and no count of tokens.
        // Its windows are bytes: 128 two-byte characters are 256 bytes,
        // sixteen windows of sixteen, the same quarter. Read by its
        // characters the request would be eight windows, and four of them
        // half of it.
        let mut text = facts(8);
        text.hash_kind = HashKind::Text;
        (text.prompt_tokens, text.prompt_chars, text.prompt_bytes) = (0, 128, 256);
        assert_eq!(scores(&slice, &text, &held(4)), vec![0.0, 0.0, 1.0]);
    }

    /// A diverting hit reserves its thin worker on the lock turn that found
    /// it eligible: the window is on the slot before the lock is let go.
    #[test]
    fn a_diverting_hit_reserves_its_thin_worker_on_the_same_turn() {
        let slice = warmup(0.25, 4);
        let pool = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
        let mut now = facts(16);
        now.now_micros = 1_000;
        for _ in 0..3 {
            assert_eq!(scores(&slice, &now, &pool), vec![1.0, 1.0, 1.0]);
            assert!(slice.divert_until.lock().is_empty());
        }
        assert_eq!(scores(&slice, &now, &pool), vec![0.0, 0.0, 1.0]);
        assert_eq!(
            slice.divert_until.lock().get(&pool[2].0.id).copied(),
            Some(1_000 + DIVERT_WINDOW_MICROS),
            "the diverted worker holds its window from the turn that picked it"
        );
    }

    #[test]
    fn a_young_fleet_slices_nothing_and_a_young_worker_among_warm_ones_takes_its_share() {
        let slice = warmup(1.0, 0);
        let (row, held) = sized(2, 0, 0, 2);
        let young = (
            row,
            AffinityFacts {
                admitted_age_ms: 1_000,
                indexed_growth: 2,
                ..held
            },
        );
        let all_young = [young, young, young];
        assert_eq!(scores(&slice, &facts(16), &all_young), vec![1.0, 1.0, 1.0]);
        // No size fed (a fleet below the level that makes anyone thin), one
        // young worker: the age rule alone picks it on every miss at share 1.
        let mixed = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), young];
        assert_eq!(scores(&slice, &facts(16), &mixed), vec![0.0, 0.0, 1.0]);
        // A hit (a deep match on another worker) is not sliced.
        let hit = [sized(0, 1, 12, 0), sized(1, 1, 0, 0), young];
        assert_eq!(scores(&slice, &facts(16), &hit), vec![1.0, 1.0, 1.0]);
    }

    #[test]
    fn the_median_sets_the_thin_boundary_when_one_index_is_large() {
        let slice = warmup(1.0, 0);
        let pool = [sized(0, 0, 0, 32), sized(1, 0, 0, 64), sized(2, 0, 0, 1024)];
        assert_eq!(scores(&slice, &facts(16), &pool), vec![1.0; 3]);
        assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn one_hit_in_divert_every_goes_to_a_thin_worker_within_its_window() {
        let slice = warmup(0.25, 4);
        // A deep hit on worker 0 (12 of 16 blocks); worker 2 is thin and idle.
        let pool = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
        let mut now = facts(16);
        now.now_micros = 1_000;
        let diverted: Vec<bool> = (0..8)
            .map(|_| scores(&slice, &now, &pool) == vec![0.0, 0.0, 1.0])
            .collect();
        // A deep hit is due once a window of divert_every hits passed
        // without a shallow one: the fourth hit diverts, and the fourth
        // after it.
        assert_eq!(
            diverted,
            vec![false, false, false, true, false, false, false, true]
        );
        // Within the window a thin worker with a request in flight takes no
        // second hit; past the window it does.
        let busy = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 1, 0, 0)];
        for _ in 0..8 {
            assert_eq!(scores(&slice, &now, &busy), vec![1.0, 1.0, 1.0]);
        }
        now.now_micros = 1_000 + DIVERT_WINDOW_MICROS;
        let later: Vec<bool> = (0..8)
            .map(|_| scores(&slice, &now, &busy) == vec![0.0, 0.0, 1.0])
            .collect();
        assert!(later.iter().any(|&d| d), "{later:?}");
        // A shallow hit (7 of 16: over the miss share, at most half) diverts
        // at divert_every hits.
        let fresh = warmup(0.25, 4);
        let shallow = [sized(0, 2, 7, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
        let hits: Vec<bool> = (0..4)
            .map(|_| scores(&fresh, &now, &shallow) == vec![0.0, 0.0, 1.0])
            .collect();
        assert_eq!(hits, vec![false, false, false, true]);
        // The worker's window leaves with it.
        fresh.on_membership(&Membership::Removed(WorkerId(3)));
        assert!(fresh.divert_until.lock().is_empty());
    }

    #[test]
    fn a_maximal_diversion_period_does_not_overflow() {
        let slice = warmup(0.25, (1_u64 << 63) + 1);
        let shallow = [sized(0, 2, 7, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
        for _ in 0..2 {
            assert_eq!(scores(&slice, &facts(16), &shallow), vec![1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn a_deep_hit_is_not_diverted_to_a_thin_worker_busier_than_its_holder() {
        let slice = warmup(0.25, 1);
        // The holder has nothing in flight; the thin worker has one: a deep
        // hit stays, a shallow one moves.
        let deep = [sized(0, 0, 12, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
        let mut now = facts(16);
        now.now_micros = 5_000_000;
        for _ in 0..3 {
            assert_eq!(scores(&slice, &now, &deep), vec![1.0, 1.0, 1.0]);
        }
        let shallow = [sized(0, 0, 7, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
        assert_eq!(scores(&slice, &now, &shallow), vec![0.0, 0.0, 1.0]);
        // A holder with more in flight than the thin worker gives its due
        // deep hit up: the holder's count is read as the thin worker's is.
        let busier = [sized(0, 4, 12, 64), sized(1, 0, 0, 64), sized(2, 1, 0, 0)];
        assert_eq!(scores(&warmup(0.25, 1), &now, &busier), vec![0.0, 0.0, 1.0]);
    }

    /// A thin worker among a deep hit's deepest holders recomputes nothing:
    /// the due hit goes to it however busy it stands beside the other
    /// holders, on either feed.
    #[test]
    fn a_thin_worker_among_the_deepest_holders_takes_a_due_hit_however_busy() {
        // Rows 0 and 2 hold the deepest match; row 2 is thin and the busier.
        let pool = [sized(0, 0, 12, 64), sized(1, 0, 0, 64), sized(2, 3, 12, 12)];
        let sizes = column(&pool);
        let holders = [holder(0, 12), holder(2, 12)];
        let by_column = RequestFacts {
            affinity: &sizes,
            now_micros: 5_000_000,
            ..facts(16)
        };
        let by_named = RequestFacts {
            named: Some(named(&holders, &[2], true)),
            now_micros: 5_000_000,
            ..facts(16)
        };
        for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
            let mut out = [0.0; 3];
            warmup(0.25, 1).score(facts, &rows(&pool), &[0, 1, 2], &mut out);
            assert_eq!(out, [0.0, 0.0, 1.0], "{feed}");
        }
    }

    /// The stages read the column and never the row: a decision fed no
    /// column reads as a miss for every row, and leaves the warm-up stage
    /// neutral before any counter moves, the answer its pass gives over rows
    /// that all read as just admitted.
    #[test]
    fn no_column_is_a_miss_for_every_row_and_slices_nothing() {
        let pool = [candidate(0, 1, 8), candidate(1, 1, 8), candidate(2, 1, 8)];
        let (held_rows, held_column) = (rows(&pool), column(&pool));
        let stage = affinity(None);
        assert_eq!(
            scores_beside(&stage, &facts(16), &held_rows, &held_column),
            vec![1.0, 1.0, 1.0]
        );
        assert_eq!(
            scores_beside(&stage, &facts(16), &held_rows, &[]),
            vec![0.5, 0.5, 0.5],
            "no column: a miss for every row, the spill gate alone orders"
        );
        // The warm-up stage: a thin worker takes a miss beside its column,
        // and nothing is sliced without one.
        let slice = warmup(1.0, 0);
        let thin = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
        let (thin_rows, thin_column) = (rows(&thin), column(&thin));
        assert_eq!(
            scores_beside(&slice, &facts(16), &thin_rows, &[]),
            vec![1.0, 1.0, 1.0]
        );
        assert_eq!(
            scores_beside(
                &slice,
                &facts(16),
                &thin_rows,
                &[AffinityFacts::default(); 3]
            ),
            vec![1.0, 1.0, 1.0],
            "rows that all read as just admitted are a young fleet"
        );
        assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
        assert_eq!(
            scores_beside(&slice, &facts(16), &thin_rows, &thin_column),
            vec![0.0, 0.0, 1.0]
        );
        assert_eq!(slice.misses.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn the_gates_read_their_parameters_and_their_defaults() {
        let gates = Gates::of(None, None, None);
        assert_eq!(gates, Gates::default());
        assert!(!gates.kv_configured());
        assert_eq!(gates.margin(0.0, 0.0), 4.0);
        assert_eq!(gates.margin(100.0, 0.0), 20.0);
        assert_eq!(gates.margin(10_000.0, 0.0), 64.0);
        let params = PrefixAffinityParams {
            spill: Some(SpillParams {
                abs: Some(8),
                rel: Some(2.0),
                prefix_margin_tokens: Some(16),
                uncached_credit: None,
            }),
            ..PrefixAffinityParams::default()
        };
        let kv = KvPressureParams {
            spread: Some(0.25),
            ..KvPressureParams::default()
        };
        let tuned = Gates::of(Some(&params), Some(&kv), Some(32));
        assert_eq!(tuned.abs, 8);
        assert_eq!(tuned.rel, 2.0);
        assert_eq!(tuned.prefix_margin_tokens, 16);
        assert_eq!(tuned.kv_spread, 0.25);
        assert_eq!(tuned.kv_ceiling, 1.0);
        assert_eq!(tuned.block_size, 32);
        assert!(tuned.kv_configured());
        // Eight blocks of 32 tokens at 16 a request price at 16 requests,
        // capped at the configured 8.
        let tokens = RequestFacts::default();
        assert_eq!(tuned.margin(0.0, tuned.tokens_of(8, &tokens)), 8.0);
        let text = RequestFacts {
            hash_kind: HashKind::Text,
            ..RequestFacts::default()
        };
        assert_eq!(tuned.tokens_of(8, &text), 64.0);
        // The block size the host counted with (an index's learned 128)
        // overrides the view's: eight blocks are 1024 tokens, and a hashless
        // request of 1024 tokens is eight blocks, not the view's 32.
        let learned = RequestFacts {
            block_size: 128,
            prompt_tokens: 1024,
            ..RequestFacts::default()
        };
        assert_eq!(tuned.tokens_of(8, &learned), 1024.0);
        assert_eq!(
            tuned.request_blocks(&learned),
            8,
            "1024 tokens at the learned 128 a block, not 32 at the view's 32"
        );
    }

    /// One index, one worker: with the pool's FIRST rows dropped, each kept
    /// row scores by its own entry of the column, not by the entry at its
    /// place among the kept.
    #[test]
    fn a_narrowed_pool_scores_each_kept_row_by_its_own_entry() {
        // The dropped rows hold the deepest matches; of the two kept, the
        // last holds more.
        let pool = [
            candidate(0, 1, 8),
            candidate(1, 1, 6),
            candidate(2, 1, 2),
            candidate(3, 1, 4),
        ];
        let column = column(&pool);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        let mut out = [0.0; 2];
        affinity(None).score(&facts, &rows(&pool), &[2, 3], &mut out);
        assert_eq!(out, [0.5, 1.0]);
    }

    /// The same for the slice: with the pool's FIRST row dropped, the thin
    /// worker among the kept is found by its own entry.
    #[test]
    fn a_narrowed_pool_finds_the_thin_worker_by_its_own_entry() {
        let params = WarmupSliceParams {
            share: Some(1.0),
            divert_every: Some(0),
            blocks: Some(8),
            miss_share: Some(0.0),
            ..WarmupSliceParams::default()
        };
        let slice = WarmupSlice::new(&params, Gates::of(None, None, Some(16)));
        // Thin at the first and at the last index; the first is dropped.
        let pool = [
            sized(0, 5, 0, 0),
            sized(1, 3, 0, 64),
            sized(2, 2, 0, 64),
            sized(3, 4, 0, 0),
        ];
        let column = column(&pool);
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        let mut out = [0.0; 3];
        slice.score(&facts, &rows(&pool), &[1, 2, 3], &mut out);
        assert_eq!(out, [0.0, 0.0, 1.0]);
    }

    /// A decision fed no column reads as one whose column holds nothing for
    /// every candidate: the whole score vector of each stage is the same,
    /// and it is the vector the stage wrote when the five facts were fields
    /// of the row and every one of them was zero.
    #[test]
    fn an_empty_column_scores_as_a_column_of_nothing_held() {
        let rows: Vec<Candidate> = (0..)
            .zip([3, 0, 7, 1, 4])
            .map(|(position, in_flight)| Candidate {
                position,
                id: WorkerId(u64::from(position) + 1),
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                in_flight,
                ..Candidate::default()
            })
            .collect();
        let nothing = vec![AffinityFacts::default(); rows.len()];
        let prefix = affinity(None);
        let slice = warmup(1.0, 0);
        let stages: [(&dyn Scorer, [f32; 5]); 3] = [
            (&prefix, [0.5; 5]),
            (&KvPressure, [1.0; 5]),
            (&slice, [1.0; 5]),
        ];
        for (stage, over_zero_fields) in stages {
            let empty = scores_beside(stage, &facts(16), &rows, &[]);
            let zeros = scores_beside(stage, &facts(16), &rows, &nothing);
            assert_eq!(empty, zeros, "{}", stage.name());
            assert_eq!(empty, over_zero_fields, "{}", stage.name());
        }
    }

    /// The warm-up stage takes one entry a candidate. A column that ends
    /// before the candidates is a host's mistake and fails the stage, as an
    /// index past its end does: walked as it stands it would read as a
    /// smaller fleet.
    #[test]
    #[should_panic(expected = "out of range")]
    fn the_warmup_stage_fails_on_a_column_that_ends_before_the_candidates() {
        let pool = [sized(0, 0, 0, 4096), sized(1, 0, 0, 4096)];
        let column = column(&pool[..1]);
        scores_beside(&warmup(1.0, 0), &facts(16), &rows(&pool), &column);
    }

    /// What a stage reads of the host it declares under its own name. The
    /// affinity stage reads the column on every source: the predictor's
    /// placements are entries of it, and that source asks for no events. The
    /// warm-up stage reads it too. The KV gate reads loads alone.
    #[test]
    fn every_reader_of_the_column_declares_it_whatever_feeds_it() {
        for source in [
            AffinitySource::Index,
            AffinitySource::Predictor,
            AffinitySource::Merged,
        ] {
            let params = PrefixAffinityParams {
                source: Some(source),
                ..PrefixAffinityParams::default()
            };
            let needs =
                PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16))).needs();
            assert!(needs.affinity, "{source:?} reads the column");
            assert_eq!(
                needs.events,
                source != AffinitySource::Predictor,
                "{source:?}: events feed the index, they do not name the column"
            );
            let placements = source != AffinitySource::Index;
            assert_eq!(
                (needs.sent_prefixes, needs.text_hashes, needs.text),
                (placements, placements, false),
                "{source:?}: text windows are read by the record, and none asks for the text"
            );
        }
        assert!(warmup(1.0, 0).needs().affinity);
        assert!(!KvPressure.needs().affinity);
    }

    /// The two stages that read the prompt's size declare it, under every
    /// source: a host may state zero where no stage declares the size
    /// ([`Needs::prompt_size`]), and a zero size reads every sent prefix as
    /// the whole prompt. The classic policy is handed the prompt's length
    /// always (`model_gateway/src/policies/cache_aware.rs:3474-3482`).
    #[test]
    fn the_stages_that_read_the_prompts_size_declare_it() {
        for source in [
            AffinitySource::Index,
            AffinitySource::Predictor,
            AffinitySource::Merged,
        ] {
            let params = PrefixAffinityParams {
                source: Some(source),
                ..PrefixAffinityParams::default()
            };
            let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
            assert!(stage.needs().prompt_size, "{source:?}");
        }
        assert!(warmup(1.0, 0).needs().prompt_size);
        assert!(!KvPressure.needs().prompt_size);
    }

    /// `row` named as holding `blocks` by the index.
    fn holder(row: u32, blocks: u32) -> AffinityHolder {
        AffinityHolder {
            row,
            overlap_blocks: blocks,
            predicted_blocks: 0,
        }
    }

    /// The rows a host names: the `holders`, and the `warming` rows, thin
    /// or young.
    fn named<'a>(
        holders: &'a [AffinityHolder],
        warming: &'a [u32],
        thin: bool,
    ) -> AffinityRows<'a> {
        AffinityRows {
            holders,
            warming,
            thin,
            mean_in_flight: None,
        }
    }

    /// The scores of `rows` beside the rows a host named and no column.
    fn scores_named(
        scorer: &dyn Scorer,
        facts: &RequestFacts<'_>,
        rows: &[Candidate],
        named: AffinityRows<'_>,
    ) -> Vec<f32> {
        // The rows' counts are lent beside them, as [`scores_beside`] lends
        // them.
        let mut rows = rows.to_vec();
        let lent = Lent::take(&mut rows);
        let facts = lent.beside(&RequestFacts {
            named: Some(named),
            ..*facts
        });
        let kept = kept(rows.len() as u32);
        let mut out = vec![0.0; rows.len()];
        scorer.score(&facts, &rows, &kept, &mut out);
        out
    }

    /// Under no cap the two feeds are still one reading: a profile that
    /// takes no sample is named every holder and every warming row. Over a
    /// pool where thirty-three of forty rows hold the request, more than
    /// any sample names, and two rows are thin, each stage scores the rows
    /// named as the column they were derived from, turn by turn.
    #[test]
    fn every_holder_named_scores_as_the_column_past_any_cap() {
        // Rows 0 to 23 hold the deepest match and rows 24 to 31 a shallower
        // one; rows 38 and 39 are thin, and row 38 holds the deepest too.
        let pool: Vec<Held> = (0..40u32)
            .map(|row| match row {
                0..=23 => sized(row, row % 3, 12, 64),
                24..=31 => sized(row, row % 3, 4, 64),
                38 => sized(row, 1, 12, 12),
                39 => sized(row, 0, 0, 12),
                _ => sized(row, row % 3, 0, 64),
            })
            .collect();
        let each: Vec<AffinityHolder> = pool
            .iter()
            .filter(|(_, held)| held.overlap_blocks > 0)
            .map(|(row, held)| holder(row.position, held.overlap_blocks))
            .collect();
        assert_eq!(each.len(), 33);
        let stage = affinity(None);
        assert_eq!(
            scores_named(
                &stage,
                &facts(16),
                &rows(&pool),
                named(&each, &[38, 39], true)
            ),
            scores(&stage, &facts(16), &pool)
        );
        let (by_column, by_named) = (warmup(0.25, 1), warmup(0.25, 1));
        for turn in 0..8 {
            assert_eq!(
                scores_named(
                    &by_named,
                    &facts(16),
                    &rows(&pool),
                    named(&each, &[38, 39], true)
                ),
                scores(&by_column, &facts(16), &pool),
                "turn {turn}"
            );
        }
    }

    /// The two feeds are one reading: over one pool, the rows a host named
    /// (each holder with what it holds, the warming rows) score as the
    /// column they were derived from, turn by turn and whichever rows a
    /// filter kept.
    #[test]
    fn rows_a_host_named_score_as_the_column_they_were_derived_from() {
        // Two holders of the deepest match, a shallower holder, a miss.
        let pool = [
            candidate(0, 1, 8),
            candidate(1, 1, 2),
            candidate(2, 1, 8),
            candidate(3, 1, 0),
        ];
        let each = [holder(0, 8), holder(1, 2), holder(2, 8)];
        let stage = affinity(None);
        let by_column = scores(&stage, &facts(16), &pool);
        assert_eq!(by_column, vec![1.0, 0.5, 1.0, 0.5]);
        assert_eq!(
            scores_named(&stage, &facts(16), &rows(&pool), named(&each, &[], false)),
            by_column
        );
        // A filter drops the deepest holders (a retry's tried rows): the
        // shallower holder is the deepest kept and leads on either feed.
        let over = |facts: RequestFacts<'_>| {
            let mut out = vec![0.0; 2];
            stage.score(&facts, &rows(&pool), &[1, 3], &mut out);
            out
        };
        let entries = column(&pool);
        let by_column = over(RequestFacts {
            affinity: &entries,
            ..facts(16)
        });
        assert_eq!(by_column, vec![1.0, 0.5]);
        assert_eq!(
            over(RequestFacts {
                named: Some(named(&each, &[], false)),
                ..facts(16)
            }),
            by_column
        );
        // A miss names no holder.
        let miss = [candidate(0, 1, 0), candidate(1, 1, 0)];
        assert_eq!(
            scores_named(&stage, &facts(16), &rows(&miss), named(&[], &[], false)),
            scores(&stage, &facts(16), &miss)
        );
        // A placement the predictor made rides a named row as it rides a
        // column's entry, for a source that reads it.
        let params = PrefixAffinityParams {
            threshold: Some(0.5),
            source: Some(AffinitySource::Merged),
            ..PrefixAffinityParams::default()
        };
        let merged = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
        let placed = |(row, held): Held, predicted_blocks: u32| {
            (
                row,
                AffinityFacts {
                    predicted_blocks,
                    ..held
                },
            )
        };
        let pool = [
            placed(candidate(0, 0, 0), 4),
            placed(candidate(1, 0, 0), 9),
            candidate(2, 0, 1),
        ];
        let each = [
            AffinityHolder {
                row: 0,
                overlap_blocks: 0,
                predicted_blocks: 4,
            },
            AffinityHolder {
                row: 1,
                overlap_blocks: 0,
                predicted_blocks: 9,
            },
            holder(2, 1),
        ];
        let by_column = scores(&merged, &facts(16), &pool);
        assert_eq!(by_column, vec![0.5, 1.0, 0.5]);
        assert_eq!(
            scores_named(&merged, &facts(16), &rows(&pool), named(&each, &[], false)),
            by_column
        );
        // One thin worker: a miss in a period goes to it on either feed.
        let thin = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
        let (by_column, by_named) = (warmup(0.25, 0), warmup(0.25, 0));
        for turn in 0..8 {
            assert_eq!(
                scores_named(&by_named, &facts(16), &rows(&thin), named(&[], &[2], true)),
                scores(&by_column, &facts(16), &thin),
                "a miss, turn {turn}"
            );
        }
        assert_eq!(by_named.misses.load(Ordering::Relaxed), 8);
        // Three thin workers: the miss goes to the least loaded one, the
        // middle of the warming rows named.
        let thin = [
            sized(0, 3, 0, 64),
            sized(1, 2, 0, 64),
            sized(2, 2, 0, 64),
            sized(3, 4, 0, 64),
            sized(4, 6, 0, 0),
            sized(5, 1, 0, 0),
            sized(6, 7, 0, 0),
        ];
        let (by_column, by_named) = (warmup(0.25, 0), warmup(0.25, 0));
        let mut taken = 0;
        for turn in 0..8 {
            let scored = scores_named(
                &by_named,
                &facts(16),
                &rows(&thin),
                named(&[], &[4, 5, 6], true),
            );
            assert_eq!(
                scored,
                scores(&by_column, &facts(16), &thin),
                "three thin rows, turn {turn}"
            );
            taken += usize::from(scored == vec![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        }
        assert!(taken > 0, "the middle warming row takes a miss");
        // A deep hit on worker 0 is diverted to the thin worker on the same
        // turns.
        let hit = [sized(0, 2, 12, 64), sized(1, 1, 0, 64), sized(2, 0, 0, 0)];
        let (by_column, by_named) = (warmup(0.25, 4), warmup(0.25, 4));
        let mut now = facts(16);
        now.now_micros = 1_000;
        let mut diverted = 0;
        for turn in 0..8 {
            let scored = scores_named(
                &by_named,
                &now,
                &rows(&hit),
                named(&[holder(0, 12)], &[2], true),
            );
            assert_eq!(scored, scores(&by_column, &now, &hit), "a hit, turn {turn}");
            diverted += usize::from(scored == vec![0.0, 0.0, 1.0]);
        }
        assert_eq!(diverted, 2);
        // Young rows among warm ones take the misses and no diverted hit.
        let young = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), sized(2, 0, 0, 2)];
        let slice = warmup(1.0, 1);
        assert_eq!(
            scores_named(&slice, &facts(16), &rows(&young), named(&[], &[2], false)),
            vec![0.0, 0.0, 1.0]
        );
        assert_eq!(
            scores_named(
                &slice,
                &facts(16),
                &rows(&young),
                named(&[holder(0, 12)], &[2], false)
            ),
            vec![1.0, 1.0, 1.0]
        );
    }

    /// Kept rows a narrowing stage swapped: a pass's walk starts over at a
    /// row behind its cursor, so each kept row scores by its own entry on
    /// either feed, at the prefix stage and at the slice.
    #[test]
    fn kept_rows_out_of_order_score_alike_on_either_feed() {
        let pool = [
            candidate(0, 1, 8),
            candidate(1, 1, 6),
            candidate(2, 1, 2),
            candidate(3, 1, 4),
        ];
        let depths = column(&pool);
        let holders = [holder(0, 8), holder(1, 6), holder(2, 2), holder(3, 4)];
        let by_column = RequestFacts {
            affinity: &depths,
            ..facts(16)
        };
        let by_named = RequestFacts {
            named: Some(named(&holders, &[], false)),
            ..facts(16)
        };
        for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
            let mut out = [0.0; 3];
            affinity(None).score(facts, &rows(&pool), &[3, 1, 2], &mut out);
            assert_eq!(out, [0.5, 1.0, 0.5], "{feed}");
        }
        // Two thin workers, the less loaded behind the other among the kept.
        let pool = [
            sized(0, 5, 0, 64),
            sized(1, 1, 0, 0),
            sized(2, 2, 0, 64),
            sized(3, 4, 0, 0),
        ];
        let sizes = column(&pool);
        let by_column = RequestFacts {
            affinity: &sizes,
            ..facts(16)
        };
        let by_named = RequestFacts {
            named: Some(named(&[], &[1, 3], true)),
            ..facts(16)
        };
        for (facts, feed) in [(&by_column, "the column"), (&by_named, "the rows named")] {
            let mut out = [0.0; 3];
            warmup(1.0, 0).score(facts, &rows(&pool), &[3, 1, 2], &mut out);
            assert_eq!(out, [0.0, 1.0, 0.0], "{feed}");
        }
    }

    /// Two arms of the rule that tells who warms and who takes a miss: a
    /// fleet fed no index size still warms its young worker, and a warming
    /// worker over the load gate takes no miss.
    #[test]
    fn a_young_worker_warms_with_no_size_fed_and_none_over_the_load_gate_takes_a_miss() {
        let (row, held) = sized(2, 0, 0, 0);
        let young = (
            row,
            AffinityFacts {
                admitted_age_ms: 1_000,
                ..held
            },
        );
        let unfed = [sized(0, 1, 0, 0), sized(1, 1, 0, 0), young];
        assert_eq!(
            scores(&warmup(1.0, 0), &facts(16), &unfed),
            vec![0.0, 0.0, 1.0]
        );
        // One thin worker beside nine warm ones: level with them it takes
        // the miss alone, far over the load gate it takes none.
        let fleet = |load: u32| {
            let mut pool: Vec<Held> = (0..9).map(|at| sized(at, 10, 0, 64)).collect();
            pool.push(sized(9, load, 0, 0));
            pool
        };
        let mut alone = vec![0.0; 10];
        alone[9] = 1.0;
        assert_eq!(scores(&warmup(1.0, 0), &facts(16), &fleet(10)), alone);
        assert_eq!(
            scores(&warmup(1.0, 0), &facts(16), &fleet(1_000)),
            vec![1.0; 10]
        );
    }

    /// A host that names no warming row ruled that nobody warms: the stage
    /// is neutral before any counter moves, hit or miss.
    #[test]
    fn no_warming_row_named_slices_nothing_and_moves_no_counter() {
        let slice = warmup(1.0, 1);
        let pool = [candidate(0, 3, 0), candidate(1, 2, 0), candidate(2, 5, 0)];
        for holders in [&[][..], &[holder(1, 9)][..]] {
            assert_eq!(
                scores_named(&slice, &facts(16), &rows(&pool), named(holders, &[], true)),
                vec![1.0, 1.0, 1.0]
            );
        }
        assert_eq!(
            (
                slice.misses.load(Ordering::Relaxed),
                slice.hits.load(Ordering::Relaxed)
            ),
            (0, 0)
        );
    }

    /// Behind a draw the rows handed are the drawn rows and the rows a host
    /// named. The gate's mean is the drawn rows' alone: a busy holder lifts
    /// a mean taken over the rows handed, and passes the gate it is judged
    /// by.
    #[test]
    fn the_spill_gate_reads_the_drawn_rows_mean_where_the_engine_set_one() {
        // The holder has 8 in flight and the two drawn rows 2 each. Over the
        // rows handed the mean is 4: 8 is not past 4 + 4, the holder stays.
        // Over the drawn rows it is 2: 8 is past 1.5 times 2 and past 2 + 4,
        // the holder spills.
        let pool = rows(&[candidate(0, 8, 0), candidate(1, 2, 0), candidate(2, 2, 0)]);
        let stage = affinity(None);
        let holders = [holder(0, 8)];
        let held = named(&holders, &[], false);
        assert_eq!(
            scores_named(&stage, &facts(16), &pool, held),
            vec![1.0, 0.5, 0.5]
        );
        let behind_a_draw = AffinityRows {
            mean_in_flight: Some(2.0),
            ..held
        };
        assert_eq!(
            scores_named(&stage, &facts(16), &pool, behind_a_draw),
            vec![0.0, 0.5, 0.5]
        );
        // The slice's gate reads the same mean: a warming row that busy
        // takes a miss by the mean of the rows handed, and none by the
        // drawn rows'.
        let warming = named(&[], &[0], true);
        assert_eq!(
            scores_named(&warmup(1.0, 0), &facts(16), &pool, warming),
            vec![1.0, 0.0, 0.0]
        );
        let behind_a_draw = AffinityRows {
            mean_in_flight: Some(2.0),
            ..warming
        };
        let slice = warmup(1.0, 0);
        assert_eq!(
            scores_named(&slice, &facts(16), &pool, behind_a_draw),
            vec![1.0, 1.0, 1.0]
        );
        assert_eq!(slice.misses.load(Ordering::Relaxed), 0);
    }

    /// Behind a draw the rows handed stand in the draw's order, and both
    /// stages read the rows a host named by a search: no pass looks behind
    /// a cursor, where the walk of a whole read asked in that order does,
    /// and the two read the same scores.
    #[test]
    fn behind_a_draw_the_stages_search_the_rows_named_and_keep_no_cursor() {
        use crate::engine::view::LOOKS_BEHIND;
        let pool: Vec<Held> = (0..6).map(|row| candidate(row, 2, 0)).collect();
        let pool = rows(&pool);
        // A draw's order. The holders are shallow, so the slice takes the
        // miss and asks its warming rows.
        let order = [4u32, 1, 5, 0, 3, 2];
        let holders = [holder(1, 4), holder(4, 4), holder(5, 3)];
        let whole = named(&holders, &[0, 3], true);
        // The mean of the rows handed, so both reads judge by one mean.
        let drawn = AffinityRows {
            mean_in_flight: Some(2.0),
            ..whole
        };
        let (prefix, slice) = (affinity(None), warmup(1.0, 0));
        for stage in [&prefix as &dyn Scorer, &slice] {
            let [by_walk, by_search] = [whole, drawn].map(|named| {
                LOOKS_BEHIND.with(|looks| looks.set(0));
                let facts = RequestFacts {
                    named: Some(named),
                    ..facts(16)
                };
                let mut out = vec![0.0; order.len()];
                stage.score(&facts, &pool, &order, &mut out);
                (out, LOOKS_BEHIND.with(std::cell::Cell::get))
            });
            assert_eq!(by_search.0, by_walk.0, "{}", stage.name());
            assert!(by_walk.1 > 0, "{}: a walk looks behind", stage.name());
            assert_eq!(by_search.1, 0, "{}: a search keeps no cursor", stage.name());
        }
    }

    /// The stage's word of a row: its match by the stage's source, of the
    /// request's blocks, and where the match was read.
    #[test]
    fn the_stage_says_what_a_row_holds_by_its_source() {
        use AffinitySource::{Index, Merged, Predictor};
        let said = |source, overlap_blocks, predicted_blocks| {
            let params = PrefixAffinityParams {
                source: Some(source),
                ..PrefixAffinityParams::default()
            };
            let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), None, Some(16)));
            let column = [AffinityFacts {
                overlap_blocks,
                predicted_blocks,
                ..AffinityFacts::default()
            }];
            let facts = RequestFacts {
                affinity: &column,
                ..facts(16)
            };
            let said = stage.said(&facts, &rows(&[candidate(0, 0, 0)]), 0);
            assert!(!said.margin_held);
            (said.blocks, said.of, said.source)
        };
        // The index's blocks, at any depth.
        assert_eq!(said(Index, 3, 12), (3, 16, OverlapSource::Index));
        // The predictor's, past the stage's threshold alone.
        assert_eq!(said(Predictor, 9, 12), (12, 16, OverlapSource::Predictor));
        assert_eq!(said(Predictor, 9, 4), (0, 16, OverlapSource::None));
        // Merged: the larger of the two, the index on a tie.
        assert_eq!(said(Merged, 3, 12), (12, 16, OverlapSource::Predictor));
        assert_eq!(said(Merged, 13, 12), (13, 16, OverlapSource::Index));
        assert_eq!(said(Merged, 12, 12), (12, 16, OverlapSource::Index));
        assert_eq!(said(Merged, 0, 0), (0, 16, OverlapSource::None));
    }

    /// A row no host named and a row past a column hold nothing. A request
    /// past sixteen bits of blocks halves both counts until its own fits,
    /// and their share stands.
    #[test]
    fn a_row_nobody_named_holds_nothing_and_a_long_request_keeps_its_share() {
        let stage = affinity(None);
        let pool = rows(&[candidate(0, 0, 0), candidate(1, 0, 0)]);
        let holders = [holder(1, 5)];
        let named = RequestFacts {
            named: Some(named(&holders, &[], false)),
            ..facts(16)
        };
        let nothing = Overlap {
            of: 16,
            ..Overlap::default()
        };
        assert_eq!(stage.said(&named, &pool, 1).blocks, 5);
        assert_eq!(stage.said(&named, &pool, 0), nothing);
        let column = [AffinityFacts {
            overlap_blocks: 70_000,
            ..AffinityFacts::default()
        }];
        let long = RequestFacts {
            affinity: &column,
            prompt_tokens: 16 * 80_000,
            ..facts(16)
        };
        let said = stage.said(&long, &pool, 0);
        assert_eq!((said.blocks, said.of), (35_000, 40_000));
        assert_eq!(stage.said(&long, &pool, 9).blocks, 0);
        let longer = RequestFacts {
            affinity: &column,
            prompt_tokens: 16 * 300_000,
            ..facts(16)
        };
        let said = stage.said(&longer, &pool, 0);
        assert_eq!((said.blocks, said.of), (8_750, 37_500));
    }

    /// One holder over the swing's margin and under the priced one: the
    /// stage says the margin held. It says so once, of its own pass, of a
    /// deepest holder, and not behind [`forget_prefix_pass`].
    #[test]
    fn the_word_of_a_pass_is_one_decisions_and_one_stages() {
        let gate = || {
            affinity(Some(SpillParams {
                rel: Some(1.1),
                prefix_margin_tokens: Some(8),
                ..SpillParams::default()
            }))
        };
        let (stage, other) = (gate(), gate());
        let pool = [
            candidate(0, 10, 8),
            candidate(1, 0, 0),
            candidate(2, 0, 0),
            candidate(3, 0, 0),
        ];
        let (rows, column) = (rows(&pool), column(&pool));
        let facts = RequestFacts {
            affinity: &column,
            ..facts(16)
        };
        let pass = |stage: &PrefixAffinity| {
            let mut out = vec![0.0; 4];
            stage.score(&facts, &rows, &kept(4), &mut out);
            out
        };
        assert_eq!(pass(&stage), vec![1.0, 0.5, 0.5, 0.5]);
        let held = Overlap {
            blocks: 8,
            of: 16,
            source: OverlapSource::Index,
            margin_held: true,
        };
        assert_eq!(stage.said(&facts, &rows, 0), held);
        // Asked again, the word is spent.
        assert!(!stage.said(&facts, &rows, 0).margin_held);
        // A row that is no deepest holder was kept by no price.
        pass(&stage);
        assert!(!stage.said(&facts, &rows, 1).margin_held);
        // The first stage of a decision to judge keeps the slot: a later
        // stage's pass does not take its word.
        pass(&stage);
        pass(&other);
        assert!(stage.said(&facts, &rows, 0).margin_held);
        // One stage's pass is no word of another's.
        pass(&other);
        assert!(!stage.said(&facts, &rows, 0).margin_held);
        // A decision whose overlap the engine states begins with no word.
        pass(&stage);
        forget_prefix_pass();
        assert!(!stage.said(&facts, &rows, 0).margin_held);
    }

    /// A later pass of the same stage writes over its own word: the word
    /// left is of the stage's last pass. Over level loads the swing's
    /// margin spills nobody, so no price kept the holder.
    #[test]
    fn a_later_pass_of_a_stage_writes_over_its_own_word() {
        let stage = affinity(Some(SpillParams {
            rel: Some(1.1),
            prefix_margin_tokens: Some(8),
            ..SpillParams::default()
        }));
        let pool = |others| {
            [
                candidate(0, 10, 8),
                candidate(1, others, 0),
                candidate(2, others, 0),
                candidate(3, others, 0),
            ]
        };
        let (busy, level) = (pool(0), pool(10));
        let pass = |pool: &[Held]| scores(&stage, &facts(16), pool);
        let held = |pool: &[Held]| {
            let column = column(pool);
            let facts = RequestFacts {
                affinity: &column,
                ..facts(16)
            };
            stage.said(&facts, &rows(pool), 0).margin_held
        };
        forget_prefix_pass();
        assert_eq!(pass(&busy), vec![1.0, 0.5, 0.5, 0.5]);
        assert!(held(&busy));
        assert_eq!(pass(&busy), vec![1.0, 0.5, 0.5, 0.5]);
        assert_eq!(pass(&level), vec![1.0, 0.5, 0.5, 0.5]);
        assert!(!held(&level));
    }

    /// A pass that orders nothing under the fleet's KV gate takes back the
    /// word an earlier pass of its stage left: the price did not keep the
    /// pick behind it.
    #[test]
    fn a_pass_under_the_kv_gate_takes_the_stages_word_back() {
        let kv = KvPressureParams {
            ceiling: Some(0.9),
            ..KvPressureParams::default()
        };
        let params = PrefixAffinityParams {
            spill: Some(SpillParams {
                rel: Some(1.1),
                prefix_margin_tokens: Some(8),
                ..SpillParams::default()
            }),
            ..PrefixAffinityParams::default()
        };
        let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
        let pool = |usage| {
            [
                reporting(0, 10, 8, usage),
                reporting(1, 0, 0, 500),
                reporting(2, 0, 0, 500),
                reporting(3, 0, 0, 500),
            ]
        };
        let (calm, hot) = (pool(500), pool(950));
        let pass = |pool: &[Held]| scores(&stage, &facts(16), pool);
        let held = || {
            let column = column(&calm);
            let facts = RequestFacts {
                affinity: &column,
                ..facts(16)
            };
            stage.said(&facts, &rows(&calm), 0).margin_held
        };
        forget_prefix_pass();
        assert_eq!(pass(&calm), vec![1.0, 0.5, 0.5, 0.5]);
        assert!(held());
        assert_eq!(pass(&calm), vec![1.0, 0.5, 0.5, 0.5]);
        assert_eq!(pass(&hot), vec![1.0; 4]);
        assert!(!held());
    }

    /// The front pass takes the word back as the whole pass does: under the
    /// fleet's KV gate it says every row alike, and the price an earlier
    /// pass judged by is not said.
    #[test]
    fn a_front_pass_under_the_kv_gate_takes_the_stages_word_back() {
        let kv = KvPressureParams {
            ceiling: Some(0.9),
            ..KvPressureParams::default()
        };
        let params = PrefixAffinityParams {
            spill: Some(SpillParams {
                rel: Some(1.1),
                prefix_margin_tokens: Some(8),
                ..SpillParams::default()
            }),
            ..PrefixAffinityParams::default()
        };
        let stage = PrefixAffinity::new(&params, Gates::of(Some(&params), Some(&kv), Some(16)));
        let pool = |usage| {
            [
                reporting(0, 10, 8, usage),
                reporting(1, 0, 0, 500),
                reporting(2, 0, 0, 500),
                reporting(3, 0, 0, 500),
            ]
        };
        let (calm, hot) = (pool(500), pool(950));
        // The front pass over a pool, lent its rows' counts as the whole
        // pass's helper lends them: the scores and what the pass said.
        let front = |pool: &[Held]| {
            let column = column(pool);
            let mut lent_rows = rows(pool);
            let lent = Lent::take(&mut lent_rows);
            let facts = lent.beside(&RequestFacts {
                affinity: &column,
                ..facts(16)
            });
            let kept = kept(lent_rows.len() as u32);
            let mut out = vec![0.0; lent_rows.len()];
            let (_, said) = stage.score_front(&facts, &lent_rows, &kept, &kept, &mut out);
            (out, matches!(said, ScoreFront::Alike))
        };
        let held = || {
            let column = column(&calm);
            let facts = RequestFacts {
                affinity: &column,
                ..facts(16)
            };
            stage.said(&facts, &rows(&calm), 0).margin_held
        };
        forget_prefix_pass();
        assert!(!front(&calm).1);
        assert!(held());
        assert!(!front(&calm).1);
        assert_eq!(front(&hot), (vec![1.0; 4], true));
        assert!(!held());
    }

    /// The margin's bit is the difference of two scores: the stage that
    /// prices the held prefix puts the row in front, and the same gate
    /// with no price puts it behind. Over pools, holders, loads and
    /// orders drawn at random, on a column, on named rows and behind a
    /// draw with the engine's mean, a decode phase among them. The word
    /// is one decision's: asked a second time it says no margin.
    #[test]
    fn the_margin_held_is_the_price_alone_keeping_a_row_in_front() {
        let gate = |prefix_margin_tokens| {
            affinity(Some(SpillParams {
                rel: Some(1.1),
                prefix_margin_tokens,
                ..SpillParams::default()
            }))
        };
        let (priced, plain) = (gate(Some(8)), gate(None));
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |below: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % below
        };
        let mut held = [0u32; 3];
        for round in 0..3_000u32 {
            let n = 1 + next(24) as u32;
            let pool: Vec<Candidate> = (0..n)
                .map(|row| candidate(row, next(40) as u32, 0).0)
                .collect();
            // Every fourth round names nobody.
            let mut holders = Vec::new();
            let mut column = vec![AffinityFacts::default(); n as usize];
            for row in 0..n {
                if round % 4 != 0 && next(3) == 0 {
                    let blocks = 1 + next(16) as u32;
                    holders.push(holder(row, blocks));
                    column[row as usize].overlap_blocks = blocks;
                }
            }
            // A draw's order: some of the rows, each once.
            let mut order: Vec<u32> = (0..n).filter(|&row| row == 0 || next(4) != 0).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, next(i as u64 + 1) as usize);
            }
            let whole = named(&holders, &[], false);
            let drawn = AffinityRows {
                mean_in_flight: Some(mean_in_flight(&RequestFacts::default(), &pool, &order)),
                ..whole
            };
            // Named rows, named rows behind a draw, a column.
            let feed = round as usize % 3;
            let facts = RequestFacts {
                affinity: if feed == 2 { &column } else { &[] },
                named: [Some(whole), Some(drawn), None][feed],
                phase: if round % 5 == 0 {
                    Phase::Decode
                } else {
                    Phase::Single
                },
                ..facts(16)
            };
            let scored = |stage: &PrefixAffinity| {
                let mut out = vec![0.0; order.len()];
                stage.score(&facts, &pool, &order, &mut out);
                out
            };
            let unpriced = scored(&plain);
            for (at, &row) in order.iter().enumerate() {
                // The pass that decided is the last that ran.
                let by_price = scored(&priced);
                let kept = by_price[at] >= 1.0 && unpriced[at] <= 0.0;
                assert_eq!(
                    priced.said(&facts, &pool, row).margin_held,
                    kept,
                    "round {round}, row {row}: {holders:?}, order {order:?}"
                );
                assert!(
                    !priced.said(&facts, &pool, row).margin_held,
                    "round {round}, row {row}: asked twice"
                );
                held[feed] += u32::from(kept);
            }
        }
        assert!(
            held.iter().all(|&rows| rows > 30),
            "rows the price alone kept, a feed: {held:?}"
        );
    }

    /// The prefix stage's body behind a draw scores as the whole pass's
    /// does: over pools, holders, loads and orders drawn at random, under
    /// a plain gate, a relative one and one that prices the held prefix,
    /// a decode phase among them. Both reads judge by one mean.
    #[test]
    fn behind_a_draw_the_prefix_stage_scores_as_the_whole_pass_does() {
        let gate = |prefix_margin_tokens| {
            affinity(Some(SpillParams {
                rel: Some(1.1),
                prefix_margin_tokens,
                ..SpillParams::default()
            }))
        };
        let stages = [affinity(None), gate(None), gate(Some(8))];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = |below: u64| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (state >> 33) % below
        };
        let mut reached = [0u32; 3];
        for round in 0..3_000u32 {
            let n = 1 + next(24) as u32;
            let pool: Vec<Candidate> = (0..n)
                .map(|row| candidate(row, next(40) as u32, 0).0)
                .collect();
            // Every fourth round names nobody.
            let mut holders = Vec::new();
            for row in 0..n {
                if round % 4 != 0 && next(3) == 0 {
                    holders.push(holder(row, 1 + next(16) as u32));
                }
            }
            // A draw's order: some of the rows, each once.
            let mut order: Vec<u32> = (0..n).filter(|&row| row == 0 || next(4) != 0).collect();
            for i in (1..order.len()).rev() {
                order.swap(i, next(i as u64 + 1) as usize);
            }
            let whole = named(&holders, &[], false);
            let drawn = AffinityRows {
                mean_in_flight: Some(mean_in_flight(&RequestFacts::default(), &pool, &order)),
                ..whole
            };
            let stage = &stages[round as usize % stages.len()];
            let [by_pass, by_draw] = [whole, drawn].map(|named| {
                let facts = RequestFacts {
                    named: Some(named),
                    phase: if round % 5 == 0 {
                        Phase::Decode
                    } else {
                        Phase::Single
                    },
                    ..facts(16)
                };
                let mut out = vec![0.0; order.len()];
                stage.score(&facts, &pool, &order, &mut out);
                out
            });
            assert_eq!(
                by_draw, by_pass,
                "round {round}: {holders:?}, order {order:?}"
            );
            for (score, count) in [0.0, 0.5, 1.0].into_iter().zip(&mut reached) {
                *count += u32::from(by_pass.contains(&score));
            }
        }
        // The rounds reach every score.
        assert!(reached.iter().all(|&rounds| rounds > 300), "{reached:?}");
    }

    /// A hit the slice diverts behind a draw reads the rows named by a
    /// search too: its holders' load, its warming rows and a busy warming
    /// row's entry look behind no cursor, where a whole read asked in that
    /// order does, and the two divert to the same row.
    #[test]
    fn behind_a_draw_a_diverted_hit_searches_the_rows_named_and_keeps_no_cursor() {
        use crate::engine::view::LOOKS_BEHIND;
        // Rows 1 and 4 hold the request deep. Of the thin rows 3 and 0 are
        // busier than the holders, so the entry of each is read, and row 2
        // takes the hit.
        let pool: Vec<Held> = (0..)
            .zip([6, 5, 1, 6, 5, 1])
            .map(|(row, busy)| candidate(row, busy, 0))
            .collect();
        let pool = rows(&pool);
        // A draw's order: each of the three reads meets a row behind the
        // one before it.
        let order = [4u32, 1, 5, 3, 0, 2];
        let holders = [holder(1, 12), holder(4, 12)];
        let whole = named(&holders, &[0, 2, 3], true);
        let drawn = AffinityRows {
            mean_in_flight: Some(4.0),
            ..whole
        };
        let [by_walk, by_search] = [whole, drawn].map(|named| {
            LOOKS_BEHIND.with(|looks| looks.set(0));
            let facts = RequestFacts {
                named: Some(named),
                ..facts(16)
            };
            let mut out = vec![0.0; order.len()];
            // A stage of its own a read: a diversion reserves its row's
            // window. Its first hit is due.
            warmup(0.25, 1).score(&facts, &pool, &order, &mut out);
            (out, LOOKS_BEHIND.with(std::cell::Cell::get))
        });
        assert_eq!(by_walk.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]);
        assert_eq!(by_search.0, by_walk.0);
        assert!(by_walk.1 > 0, "the walk looks behind");
        assert_eq!(by_search.1, 0, "a search keeps no cursor");
    }

    /// A host that names a pool's warming rows enters the stage's own
    /// ruling: for fleets of every kind the rows it is handed, and the bit,
    /// are what the stage's pass rules over a column of the same sizes and
    /// ages; a young row leaves the rows named with its window; and a
    /// parameter of the stage moves the rows named.
    #[test]
    fn a_host_that_names_warming_rows_enters_the_stages_own_ruling() {
        const NOW: u64 = 600_000_000;
        let stage = warmup(0.25, 8);
        let rule = stage.warm_up_rule().expect("the stage rules who warms");
        // A row: its index size and how many seconds ago it was admitted.
        let fleets: [&[(u32, u64)]; 8] = [
            &[(40, 500), (40, 500), (4, 500), (40, 500)],
            &[(40, 500), (40, 10), (0, 10), (40, 500)],
            &[(6, 500), (2, 10), (7, 500), (3, 20)],
            &[(0, 10), (0, 20), (0, 30)],
            &[(0, 500), (0, 500)],
            &[(50, 500), (50, 500)],
            &[(40, 10), (40, 500), (5, 10)],
            &[(9, 10), (3, 10), (6, 500), (7, 500)],
        ];
        // The rows a host names at `now` from the ruling's entry at `read`.
        let read_at = |rule: &WarmUpRule, fleet: &[(u32, u64)], read: u64, now: u64| {
            let sizes: Vec<u32> = fleet.iter().map(|row| row.0).collect();
            let admitted: Vec<u64> = fleet.iter().map(|row| NOW - row.1 * 1_000_000).collect();
            let (mut scratch, mut rows) = (Vec::new(), vec![(99, 0)]);
            let warming = rule.list_warming_rows(&sizes, &admitted, read, &mut scratch, &mut rows);
            let named = rows
                .iter()
                .filter(|&&(_, until)| warming.from <= now && now < until);
            (
                named.map(|&(row, _)| row).collect::<Vec<u32>>(),
                warming.thin,
            )
        };
        let named_by =
            |rule: &WarmUpRule, fleet: &[(u32, u64)], now: u64| read_at(rule, fleet, now, now);
        let mut seen = Vec::new();
        for fleet in fleets {
            let column: Vec<AffinityFacts> = fleet
                .iter()
                .map(|&(size, age)| AffinityFacts {
                    admitted_age_ms: (age * 1_000) as u32,
                    indexed_growth: size,
                    indexed_blocks: size,
                    ..AffinityFacts::default()
                })
                .collect();
            let by_stage = stage.ruled(&column).map_or((Vec::new(), false), |warm| {
                let mut warming = stage.warming::<false>(warm, Feed::Column(&column));
                let rows: Vec<u32> = (0..column.len() as u32)
                    .filter(|&row| warming(row))
                    .collect();
                (rows, warm.thin())
            });
            let named = named_by(&rule, fleet, NOW);
            assert_eq!(named, by_stage, "{fleet:?}");
            seen.push(named);
        }
        // The fleets hold each kind: the thin, the young, and nobody.
        assert_eq!(seen[0], (vec![2], true));
        assert_eq!(seen[2], (vec![1, 3], false));
        assert_eq!(seen[3], (vec![], false));
        // A young row is named until its window ends: sixty seconds from its
        // admission.
        assert_eq!(named_by(&rule, fleets[2], 639_999_999).0, vec![1, 3]);
        assert_eq!(named_by(&rule, fleets[2], 640_000_000).0, vec![1]);
        // The rows of one entry stand for every later instant while the
        // sizes do: read then, they are the ruling's own of that instant.
        let laters = [
            1, 29_999_999, 30_000_000, 39_999_999, 40_000_000, 49_999_999, 50_000_000, 70_000_000,
        ];
        for fleet in fleets {
            for later in laters {
                let now = NOW + later;
                let stood = read_at(&rule, fleet, NOW, now);
                assert_eq!(stood, named_by(&rule, fleet, now), "{fleet:?} +{later}");
            }
        }
        // A fleet that is all young names nobody until its first row is
        // young no more, then the rest, each until its own end.
        let all_young = |later: u64| read_at(&rule, fleets[3], NOW, NOW + later).0;
        assert_eq!(all_young(29_999_999), Vec::<u32>::new());
        assert_eq!(all_young(30_000_000), vec![0, 1]);
        assert_eq!(all_young(40_000_000), vec![0]);
        assert_eq!(all_young(50_000_000), Vec::<u32>::new());
        // A parameter of the stage moves the rows named.
        let fleet = [(40, 500), (40, 500), (15, 500), (40, 500)];
        assert_eq!(named_by(&rule, &fleet, NOW).0, vec![2]);
        let narrow = WarmupSliceParams {
            thin_ratio: Some(0.25),
            blocks: Some(8),
            ..WarmupSliceParams::default()
        };
        let narrow = WarmupSlice::new(&narrow, Gates::of(None, None, Some(16)));
        let narrow = narrow.warm_up_rule().expect("the stage rules who warms");
        assert_eq!(named_by(&narrow, &fleet, NOW).0, Vec::<u32>::new());
    }

    #[test]
    fn the_affinity_stages_answer_row_by_row_unless_the_fleet_kv_gate_is_configured() {
        let kv = KvPressureParams {
            spread: Some(0.3),
            ..KvPressureParams::default()
        };
        let plain = Gates::of(None, None, Some(16));
        let gated = Gates::of(None, Some(&kv), Some(16));
        for (gates, row_wise) in [(plain, true), (gated, false)] {
            let affinity = PrefixAffinity::new(&PrefixAffinityParams::default(), gates);
            let slice = WarmupSlice::new(&WarmupSliceParams::default(), gates);
            assert_eq!(
                (affinity.row_wise(), slice.row_wise()),
                (row_wise, row_wise)
            );
        }
        assert!(KvPressure.row_wise());
    }

    /// The two ranks say the word their scores read as ([`ScoreFront::of`]),
    /// and score a row alike by either entry. Seeded pools of one to forty
    /// rows, a part of them kept: nobody, some or every row holds the request,
    /// rows stand over and under the spill gate, and some rows warm. Two builds
    /// of each stage are asked side by side, one by each entry, so the slice's
    /// counters turn alike.
    #[test]
    fn the_ranks_say_the_word_their_scores_read_as() {
        let spill = || {
            Some(SpillParams {
                abs: Some(64),
                rel: Some(1.5),
                ..SpillParams::default()
            })
        };
        let plain: [Box<dyn Scorer>; 2] = [Box::new(affinity(spill())), Box::new(warmup(0.25, 3))];
        let said: [Box<dyn Scorer>; 2] = [Box::new(affinity(spill())), Box::new(warmup(0.25, 3))];
        let mut draw = Draw::new(0x51ce);
        // The words said, a stage: alike, no row at the ceiling, one, more.
        let mut words = [[0u32; 4]; 2];
        for round in 0..4_000u64 {
            let n = 1 + draw.below(40) as u32;
            let odds = [0, 1, 2, 8][draw.below(4)];
            let pool: Vec<Held> = (0..n)
                .map(|i| {
                    let holds = odds != 0 && draw.below(odds) == 0;
                    let indexed = [0, 64][draw.below(2)];
                    sized(i, draw.below(12) as u32, if holds { 8 } else { 0 }, indexed)
                })
                .collect();
            let (rows, column) = (rows(&pool), column(&pool));
            // One row in four is left out, and never every row.
            let mut kept: Vec<u32> = (0..n).filter(|_| draw.below(4) != 0).collect();
            if kept.is_empty() {
                kept.push(draw.below(n as usize) as u32);
            }
            let facts = RequestFacts {
                affinity: &column,
                seed: round,
                ..facts(16)
            };
            for (stage, (plain, said)) in plain.iter().zip(&said).enumerate() {
                let mut by_score = vec![0.0; kept.len()];
                let mut by_front = vec![0.0; kept.len()];
                let band = plain.score(&facts, &rows, &kept, &mut by_score);
                let (front_band, word) =
                    said.score_front(&facts, &rows, &kept, &kept, &mut by_front);
                let name = plain.name();
                assert_eq!(by_front, by_score, "{name}, round {round}");
                assert_eq!(front_band, band, "{name}, round {round}");
                assert_eq!(
                    word,
                    ScoreFront::of(plain.spread(), &by_score),
                    "{name}, round {round}: {by_score:?}"
                );
                let which = match word {
                    ScoreFront::Alike => 0,
                    ScoreFront::At(0) => 1,
                    ScoreFront::At(1) => 2,
                    _ => 3,
                };
                words[stage][which] += 1;
            }
        }
        // Each stage said each of its words in some rounds: the holders'
        // stage every one, the slice alike and its one row.
        assert!(words[0].iter().all(|&rounds| rounds > 50), "{words:?}");
        assert!(words[1][0] > 50 && words[1][2] > 50, "{words:?}");
        assert_eq!((words[1][1], words[1][3]), (0, 0), "{words:?}");
        // A lone kept row that takes the request stands at the ceiling,
        // which reads alike.
        let pool = [sized(0, 3, 0, 64), sized(1, 2, 0, 64), sized(2, 5, 0, 0)];
        let (three, held) = (rows(&pool), column(&pool));
        let fed = RequestFacts {
            affinity: &held,
            ..facts(16)
        };
        let mut lone = [0.0];
        let (_, word) = warmup(0.25, 0).score_front(&fed, &three, &[2], &[2], &mut lone);
        assert_eq!((lone, word), ([1.0], ScoreFront::Alike));
        // A decision fed no column: the slice stands alike, and no row
        // holds the request.
        let unfed = facts(16);
        let (mut slices, mut holds) = ([0.0; 3], [0.0; 3]);
        let (_, word) =
            warmup(0.25, 0).score_front(&unfed, &three, &kept(3), &kept(3), &mut slices);
        assert_eq!((slices, word), ([1.0; 3], ScoreFront::Alike));
        let (_, word) = affinity(None).score_front(&unfed, &three, &kept(3), &kept(3), &mut holds);
        assert_eq!((holds, word), ([0.5; 3], ScoreFront::At(0)));
        // The rows a host named say nothing, and score as by the plain
        // entry: the second row holds the request and the third warms.
        let holders = [holder(1, 8)];
        let naming = RequestFacts {
            named: Some(named(&holders, &[2], true)),
            ..facts(16)
        };
        let pairs: [[Box<dyn Scorer>; 2]; 2] = [
            [Box::new(affinity(None)), Box::new(affinity(None))],
            [Box::new(warmup(0.25, 0)), Box::new(warmup(0.25, 0))],
        ];
        for [plain, said] in &pairs {
            let (mut by_score, mut by_front) = ([0.0; 3], [0.0; 3]);
            let band = plain.score(&naming, &three, &kept(3), &mut by_score);
            assert_eq!(
                said.score_front(&naming, &three, &kept(3), &kept(3), &mut by_front),
                (band, ScoreFront::Unsaid),
                "{}",
                plain.name()
            );
            assert_eq!(by_front, by_score, "{}", plain.name());
        }
        // Under the fleet's KV gate both ranks stand alike.
        let kv = KvPressureParams {
            spread: Some(0.3),
            ceiling: Some(0.9),
            ..KvPressureParams::default()
        };
        let params = PrefixAffinityParams::default();
        let gates = Gates::of(Some(&params), Some(&kv), Some(16));
        let slicing = WarmupSliceParams {
            share: Some(0.25),
            ..WarmupSliceParams::default()
        };
        let gated: [Box<dyn Scorer>; 2] = [
            Box::new(PrefixAffinity::new(&params, gates)),
            Box::new(WarmupSlice::new(&slicing, gates)),
        ];
        let ceiling = [reporting(0, 0, 8, 950), reporting(1, 0, 0, 900)];
        let (two, usage) = (rows(&ceiling), column(&ceiling));
        let pressed = RequestFacts {
            affinity: &usage,
            ..facts(16)
        };
        for stage in &gated {
            let mut out = [0.0; 2];
            let (_, word) = stage.score_front(&pressed, &two, &kept(2), &kept(2), &mut out);
            assert_eq!(
                (out, word),
                ([1.0, 1.0], ScoreFront::Alike),
                "{}",
                stage.name()
            );
        }
    }
}
