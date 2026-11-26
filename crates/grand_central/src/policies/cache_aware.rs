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
    tracking::lease::Membership,
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
#[non_exhaustive]
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
        for (slot, &i) in self.kept.iter().enumerate() {
            let c = &self.candidates[i as usize];
            let held = self.facts.in_flight(c);
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
            // A candidate that predicts no block holds none by this
            // source whichever way its share reads: the share is not
            // divided for it.
            AffinitySource::Predictor | AffinitySource::Merged
                if held.predicted_blocks > 0
                    && whole.clears(held.predicted_blocks, self.threshold) =>
            {
                held.predicted_blocks
            }
            AffinitySource::Predictor | AffinitySource::Merged => 0,
        };
        index.max(predicted)
    }

    /// The stage's scores over what its two passes read of the kept rows.
    /// `feed` says whether anybody holds anything and carries the drawn
    /// rows' mean where the engine set one. `reads` hands the first pass
    /// its reader: a column's is a read by index, and the rows a host
    /// named are walked beside the kept rows. A row's entry is read once:
    /// its match rides the row's score slot to the pass that scores, so
    /// `out` holds a slot for every kept row. One body a feed, the
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
        reads: impl FnOnce() -> R,
    ) -> (TieBand, u32) {
        debug_assert_eq!(out.len(), kept.len(), "one score slot a kept row");
        let whole = self.gates.whole(facts);
        let holds = !feed.is_empty();
        // Nobody holding anything, the deepest match is zero without a
        // pass; any other feed answers for every kept row.
        let mut best = 0;
        if holds {
            // Each row's entry is read once: its match rides the row's
            // score slot, bit for bit, to the pass that scores.
            let mut entry = reads();
            for (o, &i) in out.iter_mut().zip(kept) {
                let matched = self.matched(entry(i), whole);
                *o = f32::from_bits(matched);
                best = best.max(matched);
            }
        }
        let avg = spill_mean(feed.drawn_mean(), facts, candidates, kept);
        // A decode worker prefills nothing again: its held prefix is not
        // priced.
        let held = if self.priced(facts) {
            self.judge(best, avg);
            self.gates.tokens_of(best, facts)
        } else {
            0.0
        };
        let mut ceilings = 0u32;
        for (o, &i) in out.iter_mut().zip(kept) {
            let c = &candidates[i as usize];
            let load = facts.in_flight(c) as f32;
            // Where the deepest match is above zero, the pass above wrote
            // this row's match into its slot.
            *o = if best > 0 && o.to_bits() == best && !self.gates.over(load, avg, held) {
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
        for (o, &i) in out.iter_mut().zip(kept) {
            let load = facts.in_flight(&candidates[i as usize]) as f32;
            // A match above zero was read into this slot.
            let deepest = (best > 0) & (o.to_bits() == best);
            let own = usize::from(deepest & !holder.over(load));
            *o = [0.0, 0.5, 1.0, 1.0][own << 1 | usize::from(!anyone.over(load))];
        }
        TieBand(0.0)
    }

    /// The stage's scores over the rows a host named: a holder reads its
    /// depth. The first pass walks the holders beside the kept rows where
    /// the pool is read whole; behind a draw each row is searched once
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

    /// The spill gate (`Gates::spill`) moves a request off a holder that
    /// stands over it: one loaded past the pool's mean by both its bounds.
    fn balances_load_against_affinity(&self) -> bool {
        true
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
    /// one. Under the fleet's KV gate the stage ordered nothing: the row's
    /// match is said all the same, and no margin.
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
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
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

    /// The slice sends a miss only to a warming worker under the spill gate
    /// (`Gates::over`), and diverts a deep hit only to a worker with less
    /// in flight than the hit's holder.
    fn balances_load_against_affinity(&self) -> bool {
        true
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
#[path = "../../tests/unit/policies/cache_aware/tests.rs"]
mod tests;
