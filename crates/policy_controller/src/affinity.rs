//! The affinity stages of a profile: `prefix-affinity` (the holders of the
//! request's prefix, with the match threshold and the spill gate),
//! `warmup-slice` (the slice of misses and the diverted hits a worker with a
//! thin or young cache takes) and `kv-pressure` (the fleet gate under which
//! affinity yields to load). The three read one set of [`Gates`], so a
//! profile's spill and KV-pressure parameters configure all of them: the
//! `kv-pressure` stage orders nothing itself, and the spill gate the slice
//! checks its candidates against is the affinity stage's. A stateful part
//! (the slice's period, the diversion's window per worker) keeps its state
//! behind the lifecycle alone, and no stage allocates per decision.
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
    framework::{Scorer, TieBand},
    lease::{Lifecycle, Membership},
    spec::{AffinitySource, KvPressureParams, Needs, PrefixAffinityParams, WarmupSliceParams},
    view::{Candidate, HashKind, Phase, RequestFacts, WorkerId},
    Draw,
};

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

/// Characters a text-window block prices at per token, for the held prefix's
/// price under `hash_kind: text`.
const CHARS_PER_TOKEN: u32 = 4;

/// Tokens per block when the document names none: the block size of record
/// of the common engines.
const DEFAULT_BLOCK_SIZE: u32 = 16;

/// The gates the affinity stages of one profile share: the spill gate of
/// `prefix-affinity`, the fleet KV gate of `kv-pressure`, and the block size
/// a held prefix is priced by.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gates {
    /// A holder loaded past this multiple of the kept rows' mean in flight
    /// AND past the absolute margin above it spills.
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
    fn kv_imbalanced(&self, candidates: &[Candidate], kept: &[u32]) -> bool {
        if !self.kv_configured() {
            return false;
        }
        let mut bounds: Option<(f32, f32)> = None;
        for &i in kept {
            let Some(load) = candidates[i as usize].load else {
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
    /// gate at the kept rows' mean `avg`.
    fn over(&self, load: f32, avg: f32, held_tokens: f32) -> bool {
        load > avg * self.rel && load > avg + self.margin(avg, held_tokens)
    }

    /// The size of one of the request's blocks: the facts' own when the
    /// host counted with one (an index's learned size), else the view's.
    fn block_size_of(&self, facts: &RequestFacts<'_>) -> u32 {
        if facts.block_size > 0 {
            facts.block_size
        } else {
            self.block_size
        }
    }

    /// The request's size in blocks: its block hashes when the host sent
    /// them, else its token count over the block size, never under one.
    fn request_blocks(&self, facts: &RequestFacts<'_>) -> u32 {
        if facts.block_hashes.is_empty() {
            (facts.prompt_tokens / self.block_size_of(facts)).max(1)
        } else {
            u32::try_from(facts.block_hashes.len()).unwrap_or(u32::MAX)
        }
    }

    /// `blocks` of the request in tokens, for the held prefix's price: at
    /// the block size the request was counted with.
    fn tokens_of(&self, blocks: u32, facts: &RequestFacts<'_>) -> f32 {
        let units = blocks as f32 * self.block_size_of(facts) as f32;
        match facts.hash_kind {
            HashKind::Token => units,
            HashKind::Text => units / CHARS_PER_TOKEN as f32,
        }
    }
}

/// What one pass of the warm-up stage over a decision reads: the rows, the
/// fleet's index level, the request's deepest index match and size, and the
/// decision's draw.
struct Pass<'a> {
    candidates: &'a [Candidate],
    kept: &'a [u32],
    level: u32,
    best: u32,
    request_blocks: u32,
    draw: Draw,
}

impl Pass<'_> {
    /// The slot of the least-loaded kept row `wanted` admits, equal loads
    /// drawn uniformly from the decision's seed.
    fn least_loaded(&mut self, mut wanted: impl FnMut(&Candidate) -> bool) -> Option<usize> {
        let mut pick: Option<(usize, u32)> = None;
        let mut tied = 0usize;
        for (slot, &i) in self.kept.iter().enumerate() {
            let c = &self.candidates[i as usize];
            if !wanted(c) {
                continue;
            }
            match pick {
                Some((_, load)) if c.in_flight > load => {}
                Some((_, load)) if c.in_flight == load => {
                    tied += 1;
                    if self.draw.below(tied + 1) == 0 {
                        pick = Some((slot, load));
                    }
                }
                _ => {
                    pick = Some((slot, c.in_flight));
                    tied = 0;
                }
            }
        }
        pick.map(|(slot, _)| slot)
    }
}

/// The mean of the kept rows' in-flight counts.
fn mean_in_flight(candidates: &[Candidate], kept: &[u32]) -> f32 {
    if kept.is_empty() {
        return 0.0;
    }
    let sum: u64 = kept
        .iter()
        .map(|&i| u64::from(candidates[i as usize].in_flight))
        .sum();
    sum as f32 / kept.len() as f32
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
/// rows holding the deepest match and under the spill gate score `1.0`, the
/// rows under the gate `0.5`, the rest `0.0`; under the fleet KV gate every
/// row scores alike and the load scorer decides. The values are exact, so
/// the band is zero.
#[derive(Debug)]
pub struct PrefixAffinity {
    threshold: f32,
    source: AffinitySource,
    gates: Gates,
}

impl PrefixAffinity {
    /// The stage from its parameters and its profile's gates.
    pub fn new(params: &PrefixAffinityParams, gates: Gates) -> Self {
        Self {
            threshold: params.threshold.unwrap_or(0.3),
            source: params.source.unwrap_or_default(),
            gates,
        }
    }

    /// The blocks of the request `c` holds by this stage's source.
    fn matched(&self, c: &Candidate, request_blocks: u32) -> u32 {
        let index = match self.source {
            AffinitySource::Predictor => 0,
            AffinitySource::Index | AffinitySource::Merged => c.overlap_blocks,
        };
        let predicted = match self.source {
            AffinitySource::Index => 0,
            AffinitySource::Predictor | AffinitySource::Merged
                if c.predicted_blocks as f32 / request_blocks as f32 > self.threshold =>
            {
                c.predicted_blocks
            }
            AffinitySource::Predictor | AffinitySource::Merged => 0,
        };
        index.max(predicted)
    }
}

impl Lifecycle for PrefixAffinity {}

impl Scorer for PrefixAffinity {
    fn name(&self) -> &'static str {
        "prefix-affinity"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        if self.gates.kv_imbalanced(candidates, kept) {
            return neutral(out);
        }
        let request_blocks = self.gates.request_blocks(facts);
        let best = kept
            .iter()
            .map(|&i| self.matched(&candidates[i as usize], request_blocks))
            .max()
            .unwrap_or(0);
        let avg = mean_in_flight(candidates, kept);
        // A decode worker prefills nothing again: its held prefix is not
        // priced.
        let priced = self.gates.prefix_margin_tokens > 0 && facts.phase != Phase::Decode;
        let held = if priced {
            self.gates.tokens_of(best, facts)
        } else {
            0.0
        };
        for (o, &i) in out.iter_mut().zip(kept) {
            let c = &candidates[i as usize];
            let load = c.in_flight as f32;
            *o = if best > 0
                && self.matched(c, request_blocks) == best
                && !self.gates.over(load, avg, held)
            {
                1.0
            } else if !self.gates.over(load, avg, 0.0) {
                0.5
            } else {
                0.0
            };
        }
        TieBand(0.0)
    }

    fn needs(&self) -> Needs {
        let placements = self.source != AffinitySource::Index;
        Needs {
            token_hashes: true,
            events: self.source != AffinitySource::Predictor,
            text: placements,
            text_hashes: placements,
            loads: self.gates.kv_configured(),
            ..Needs::default()
        }
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
            secs_micros: params.secs.unwrap_or(60).saturating_mul(1_000_000),
            share: params.share.unwrap_or(0.25),
            blocks: u32::try_from(params.blocks.unwrap_or(1024)).unwrap_or(u32::MAX),
            thin_ratio: params.thin_ratio.unwrap_or(0.5),
            divert_every: params.divert_every.unwrap_or(8),
            miss_share: params.miss_share.unwrap_or(0.3),
            gates,
            misses: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            shallow_seen: AtomicBool::new(false),
            divert_until: Mutex::new(FxHashMap::default()),
            sizes: Mutex::new(Vec::new()),
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
        self.thin_ratio > 0.0
            && level >= self.blocks
            && (indexed == 0 || f64::from(indexed) < f64::from(level) * f64::from(self.thin_ratio))
    }

    /// Whether a worker is young: admitted within the window and short of
    /// its first blocks.
    fn is_young(&self, c: &Candidate) -> bool {
        u64::from(c.admitted_age_ms).saturating_mul(1_000) < self.secs_micros
            && c.indexed_growth < self.blocks
    }

    /// The fleet's index level: the upper median of the rows' sizes, so one
    /// emptied worker does not drag it down.
    fn fleet_level(&self, candidates: &[Candidate]) -> u32 {
        let mut sizes = self.sizes.lock();
        sizes.clear();
        sizes.extend(candidates.iter().map(|c| c.indexed_blocks));
        if sizes.is_empty() {
            return 0;
        }
        let mid = sizes.len() / 2;
        let (_, level, _) = sizes.select_nth_unstable(mid);
        *level
    }

    /// The slot of the kept row the slice sends this miss to, if it is due.
    fn slice(&self, pass: &mut Pass<'_>, thin_any: bool, young_all: bool) -> Option<usize> {
        let avg = mean_in_flight(pass.candidates, pass.kept);
        let level = pass.level;
        // A thin worker is a candidate whatever the fleet's age; workers
        // warming only because they are young are candidates only when the
        // fleet is not all young.
        let warming = |c: &Candidate| {
            if thin_any {
                self.is_thin(c.indexed_blocks, level)
            } else {
                !young_all && self.is_young(c)
            }
        };
        let slot =
            pass.least_loaded(|c| warming(c) && !self.gates.over(c.in_flight as f32, avg, 0.0))?;
        let seen = self.misses.fetch_add(1, Ordering::Relaxed);
        seen.is_multiple_of(self.period()).then_some(slot)
    }

    /// The slot of the kept row a hit is diverted to, if one is due.
    fn divert(&self, pass: &mut Pass<'_>, now: u64) -> Option<usize> {
        if self.divert_every == 0 {
            return None;
        }
        let Pass {
            candidates,
            kept,
            level,
            best,
            request_blocks,
            ..
        } = *pass;
        let shallow = best as f32 / request_blocks as f32 <= DIVERT_SHALLOW_SHARE;
        if shallow {
            self.shallow_seen.store(true, Ordering::Relaxed);
        }
        let hits = self.hits.fetch_add(1, Ordering::Relaxed) + 1;
        let shallow_seen = self.shallow_seen.load(Ordering::Relaxed);
        let due = (hits >= self.divert_every && (shallow || !shallow_seen))
            || hits >= 2 * self.divert_every;
        if !due {
            return None;
        }
        // What a deep hit's thin worker is measured against: the least
        // loaded of the request's deepest holders. A thin worker among them
        // recomputes nothing and is exempt.
        let holder_load = if shallow {
            None
        } else {
            kept.iter()
                .map(|&i| &candidates[i as usize])
                .filter(|c| c.overlap_blocks == best)
                .map(|c| c.in_flight)
                .min()
        };
        let slot = {
            let until = self.divert_until.lock();
            pass.least_loaded(|c| {
                self.is_thin(c.indexed_blocks, level)
                    && !(c.in_flight > 0 && until.get(&c.id).is_some_and(|&t| now < t))
                    && !(holder_load.is_some_and(|h| c.in_flight >= h) && c.overlap_blocks != best)
            })?
        };
        self.hits.store(0, Ordering::Relaxed);
        self.shallow_seen.store(false, Ordering::Relaxed);
        self.divert_until.lock().insert(
            candidates[kept[slot] as usize].id,
            now + DIVERT_WINDOW_MICROS,
        );
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
            Membership::Added(_) | Membership::Relabelled(_) => {}
        }
    }
}

impl Scorer for WarmupSlice {
    fn name(&self) -> &'static str {
        "warmup-slice"
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        if self.share <= 0.0 || self.gates.kv_imbalanced(candidates, kept) {
            return neutral(out);
        }
        // A settled fleet (no index size fed, nobody young) costs one pass.
        let mut any_indexed = false;
        let mut young = 0usize;
        for c in candidates {
            any_indexed |= c.indexed_blocks > 0;
            young += usize::from(self.is_young(c));
        }
        if !any_indexed && young == 0 {
            return neutral(out);
        }
        let level = if any_indexed {
            self.fleet_level(candidates)
        } else {
            0
        };
        let thin_any = candidates
            .iter()
            .any(|c| self.is_thin(c.indexed_blocks, level));
        if !thin_any && (young == 0 || young == candidates.len()) {
            return neutral(out);
        }
        let request_blocks = self.gates.request_blocks(facts);
        let best = kept
            .iter()
            .map(|&i| candidates[i as usize].overlap_blocks)
            .max()
            .unwrap_or(0);
        let thin_overlap = best as f32 / request_blocks as f32 <= self.miss_share
            || (best <= WARMUP_MISS_BLOCKS && best * 2 < request_blocks);
        let mut pass = Pass {
            candidates,
            kept,
            level,
            best,
            request_blocks,
            draw: Draw::new(facts.seed ^ 0x77a7_2d5e_6f1c_3b09),
        };
        let slot = if thin_overlap {
            self.slice(&mut pass, thin_any, young == candidates.len())
        } else if thin_any {
            self.divert(&mut pass, facts.now_micros)
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

    fn needs(&self) -> Needs {
        Needs {
            events: true,
            loads: true,
            ..Needs::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        spec::SpillParams,
        view::{Health, LoadRow},
    };

    fn candidate(position: u32, in_flight: u32, overlap_blocks: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(u64::from(position) + 1),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            overlap_blocks,
            admitted_age_ms: u32::MAX,
            ..Candidate::default()
        }
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

    fn scores(scorer: &dyn Scorer, facts: &RequestFacts<'_>, pool: &[Candidate]) -> Vec<f32> {
        let kept = kept(pool.len() as u32);
        let mut out = vec![0.0; pool.len()];
        scorer.score(facts, pool, &kept, &mut out);
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
        let mean = mean_in_flight(&pool, &kept(3));
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
        let shallow = Candidate {
            predicted_blocks: 4,
            ..candidate(0, 0, 0)
        };
        let deep = Candidate {
            predicted_blocks: 9,
            ..candidate(1, 0, 0)
        };
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

    fn reporting(position: u32, in_flight: u32, overlap_blocks: u32, active_kv: u16) -> Candidate {
        Candidate {
            load: Some(LoadRow {
                active_kv_permille: active_kv,
                kv_usage_permille: active_kv,
                live_rate: 1000,
                ..LoadRow::default()
            }),
            ..candidate(position, in_flight, overlap_blocks)
        }
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

    fn sized(position: u32, in_flight: u32, overlap_blocks: u32, indexed: u32) -> Candidate {
        Candidate {
            indexed_blocks: indexed,
            indexed_growth: indexed,
            ..candidate(position, in_flight, overlap_blocks)
        }
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

    #[test]
    fn a_young_fleet_slices_nothing_and_a_young_worker_among_warm_ones_takes_its_share() {
        let slice = warmup(1.0, 0);
        let young = Candidate {
            admitted_age_ms: 1_000,
            indexed_growth: 2,
            ..sized(2, 0, 0, 2)
        };
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
}
