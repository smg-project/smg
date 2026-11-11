//! The built-in stages: filters, scorers and pickers the presets compose.
//! Every stateful one keeps its state behind the lifecycle alone, and none
//! allocates per decision.

use std::collections::{HashMap, VecDeque};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    decision::{Target, Targets, MAX_TARGETS},
    framework::{Filter, Picker, Scorer, TieBand},
    lease::{Lease, Lifecycle, Membership, Side},
    spec::Needs,
    view::{Candidate, Constraints, LoadRow, RequestFacts, WorkerId},
    Draw,
};

/// Expected waits within this many seconds of the minimum are one tie: the
/// least-load scorer's own band, mapped onto the normalized scale per
/// decision.
const TIE_EPSILON_SECS: f32 = 1e-6;

// ---------------------------------------------------------------- filters

/// Ready, breaker closed, not stalled, not draining.
#[derive(Debug)]
pub struct Healthy;

impl Filter for Healthy {
    fn name(&self) -> &'static str {
        "healthy"
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.health.routable_ignoring_overload()
    }
}

/// Not over the overload thresholds, unless the host lifted the veto.
#[derive(Debug)]
pub struct NotOverloaded;

impl Filter for NotOverloaded {
    fn name(&self) -> &'static str {
        "not-overloaded"
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        cx.overload_lifted || !c.health.overloaded
    }
}

/// The prefill capacity window has a slot (a host-supplied bit).
#[derive(Debug)]
pub struct HasCapacity;

impl Filter for HasCapacity {
    fn name(&self) -> &'static str {
        "has-capacity"
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.has_capacity
    }
}

/// The decode's running window has a free room (a host-supplied bit). Soft:
/// with every candidate full the list stands and the dispatch waits at its
/// pick's gate, as today.
#[derive(Debug)]
pub struct HasRoom;

impl Filter for HasRoom {
    fn name(&self) -> &'static str {
        "has-room"
    }

    fn keep(&self, _: &RequestFacts<'_>, c: &Candidate, _: &Constraints<'_>) -> bool {
        c.has_room
    }

    fn soft(&self) -> bool {
        true
    }
}

/// Not tried by an earlier attempt and not excluded by the host. Soft: the
/// engines a retry already tried step aside while another remains.
#[derive(Debug)]
pub struct NotTried;

impl Filter for NotTried {
    fn name(&self) -> &'static str {
        "not-tried"
    }

    fn keep(&self, facts: &RequestFacts<'_>, c: &Candidate, cx: &Constraints<'_>) -> bool {
        !facts.tried.contains(&c.id) && !cx.excluded.contains(&c.id)
    }

    fn soft(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------- scorers

/// Lower raw costs become higher normalized scores; equal costs score `1.0`.
/// Returns the span of the costs, `0.0` when they are all equal.
fn normalize_costs(out: &mut [f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &c in out.iter() {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let span = hi - lo;
    let spread = span > 0.0 && span.is_finite();
    for o in out.iter_mut() {
        *o = if spread { 1.0 - (*o - lo) / span } else { 1.0 };
    }
    if spread {
        span
    } else {
        0.0
    }
}

/// Fewest requests in flight on the host.
#[derive(Debug)]
pub struct LeastRequests;

impl Lifecycle for LeastRequests {}

impl Scorer for LeastRequests {
    fn name(&self) -> &'static str {
        "least-requests"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = candidates[i as usize].in_flight as f32;
        }
        normalize_costs(out);
        TieBand::NORMALIZED
    }
}

/// The owner of the request's session key scores `1.0`, every other `0.0`.
#[derive(Debug)]
pub struct SessionAffinity;

impl Lifecycle for SessionAffinity {}

impl Scorer for SessionAffinity {
    fn name(&self) -> &'static str {
        "session-affinity"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = if candidates[i as usize].session_hit {
                1.0
            } else {
                0.0
            };
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            headers: true,
            ..Needs::default()
        }
    }
}

/// Prefix overlap in blocks over the request's blocks.
#[derive(Debug)]
pub struct PrefixAffinity;

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
        let blocks = facts.block_hashes.len().max(1) as f32;
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = (candidates[i as usize].overlap_blocks as f32 / blocks).clamp(0.0, 1.0);
        }
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            token_hashes: true,
            events: true,
            ..Needs::default()
        }
    }
}

/// Dispatches to one rank row no report reflects yet, oldest first.
#[derive(Debug, Default)]
struct SinceReport {
    tokens: u64,
    requests: u64,
    dispatches: VecDeque<(u64, u32)>,
}

/// Dispatches kept per rank row between reports.
const SINCE_REPORT_KEPT: usize = 8_192;

impl SinceReport {
    fn record(&mut self, at_micros: u64, tokens: u32) {
        if self.dispatches.len() >= SINCE_REPORT_KEPT {
            self.pop_oldest();
        }
        self.dispatches.push_back((at_micros, tokens));
        self.tokens += u64::from(tokens);
        self.requests += 1;
    }

    fn pop_oldest(&mut self) {
        if let Some((_, tokens)) = self.dispatches.pop_front() {
            self.tokens -= u64::from(tokens);
            self.requests -= 1;
        }
    }

    /// Release the dispatches made at or before the report's sample.
    fn release_through(&mut self, sampled_at_micros: u64) {
        while self
            .dispatches
            .front()
            .is_some_and(|&(at, _)| at <= sampled_at_micros)
        {
            self.pop_oldest();
        }
    }
}

/// A candidate row's identity in the tally: the worker and the DP rank. A row
/// is one worker or one rank of a data-parallel worker, and a rank's report
/// and credit are its own (design 4.1: a rank is a candidate), so two ranks of
/// one worker never share an entry. The id is a fingerprint already, so the
/// tally hashes the pair with the repository's fast hasher: the default
/// hasher's second block over the two-word key cost the credited decision 3
/// to 8 % at 64 and 512 rows.
type RankKey = (WorkerId, u16);

/// The expected wait: token-work over drain rate plus a convex KV-pressure
/// barrier, the since-report dispatch credit water-filling arrivals between
/// reports. The host's in-flight count floors the report. The credit tally is
/// per rank row.
#[derive(Debug)]
pub struct LeastLoad {
    pub kv_pressure_weight: f32,
    pub mean_prefill_tokens: u32,
    pub default_throughput: f32,
    tally: Mutex<FxHashMap<RankKey, SinceReport>>,
}

impl LeastLoad {
    pub fn new(kv_pressure_weight: f32, mean_prefill_tokens: u32, default_throughput: f32) -> Self {
        Self {
            kv_pressure_weight,
            mean_prefill_tokens: mean_prefill_tokens.max(1),
            default_throughput: if default_throughput > 0.0 {
                default_throughput
            } else {
                1.0
            },
            tally: Mutex::new(FxHashMap::default()),
        }
    }

    /// Dispatches no report reflects yet on `worker`, over every rank of it,
    /// for tests and dumps.
    pub fn held(&self, worker: WorkerId) -> usize {
        self.tally
            .lock()
            .iter()
            .filter(|((id, _), _)| *id == worker)
            .map(|(_, d)| d.dispatches.len())
            .sum()
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.tally
            .lock()
            .get(&(worker, dp_rank))
            .map_or(0, |d| d.dispatches.len())
    }

    /// A reporting candidate's dispatch credit as token-work: the since-report
    /// tokens plus the host's in-flight requests no report or credit accounts
    /// for, at the mean prefill. Read under the tally lock; no division here.
    fn credit_work(&self, c: &Candidate, credit_tokens: u64, credit_requests: u64) -> f32 {
        let reported = c
            .load
            .map_or(0, |r| u64::from(r.running) + u64::from(r.queued));
        let unaccounted = u64::from(c.in_flight).saturating_sub(reported + credit_requests);
        unaccounted as f32 * self.mean_prefill_tokens as f32 + credit_tokens as f32
    }

    /// The expected wait of a reporting candidate once its credit work is
    /// known: the queued and running work over the drain rate, plus the convex
    /// KV-pressure barrier over the running requests' share of the cache (a
    /// full but reclaimable prefix cache is no pressure). Runs after the lock
    /// is gone.
    fn reported_cost(&self, r: LoadRow, credit_work: f32, nominal: f32) -> f32 {
        let mean = self.mean_prefill_tokens as f32;
        let queued = if r.waiting_uncached_tokens > 0 {
            r.waiting_uncached_tokens as f32
        } else {
            r.queued as f32 * mean
        };
        let running = r.running as f32 * mean;
        let drain = if r.live_rate > 0 {
            r.live_rate as f32
        } else {
            nominal
        };
        let k = r.active_kv().clamp(0.0, 0.999_999);
        (queued + running + credit_work) / drain + self.kv_pressure_weight * k / (1.0 - k)
    }

    /// The cost of a candidate without a report: the reporting peers' baseline
    /// plus its in-flight work, or the in-flight count alone when no peer
    /// reports.
    fn unreported_cost(
        &self,
        c: &Candidate,
        nominal: f32,
        fleet_has_loads: bool,
        peer_baseline: f32,
    ) -> f32 {
        if fleet_has_loads {
            peer_baseline + c.in_flight as f32 * self.mean_prefill_tokens as f32 / nominal
        } else {
            c.in_flight as f32
        }
    }
}

/// A cost slot not written yet: a candidate without a report is costed
/// after the reporting peers' baseline is known. Costs are never negative.
const PENDING: f32 = f32::NEG_INFINITY;

impl Lifecycle for LeastLoad {
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        // The decode side of a pair target is the pair's decode rank; the
        // target itself is the prefill. The credit dates from the dispatch,
        // so a report sampled while a queue held the decision does not
        // release it.
        let rank: RankKey = match side {
            Side::Decode => target.pair.map_or((target.id, target.dp_rank), |p| {
                (p.decode_id, p.decode_dp_rank)
            }),
            Side::Single | Side::Prefill | Side::Encode => (target.id, target.dp_rank),
        };
        let dispatched_micros = lease.issued_micros + u64::from(since_decision_micros);
        let mut tally = self.tally.lock();
        tally
            .entry(rank)
            .or_default()
            .record(dispatched_micros, lease.tokens);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            // Every rank of the worker goes with it. A resized worker's ranks
            // are a new set: what a vanished rank held would never release,
            // and a kept rank's next report is one interval away. Rare, so a
            // walk over the tally is cheaper than a second index per worker.
            Membership::Removed(id)
            | Membership::Replaced { old: id, .. }
            | Membership::Resized(id) => {
                self.tally.lock().retain(|(worker, _), _| worker != id);
            }
            // A snapshot (every worker of every pool, after a resync) stands
            // for removals that may never have arrived: only its ids keep a
            // tally. Rare, so a sorted copy is cheaper than a scan per row.
            Membership::Snapshot(ids) => {
                let mut sorted: Vec<u64> = ids.iter().map(|w| w.0).collect();
                sorted.sort_unstable();
                self.tally
                    .lock()
                    .retain(|(worker, _), _| sorted.binary_search(&worker.0).is_ok());
            }
            Membership::Added(_) | Membership::Relabelled(_) => {}
        }
    }
}

impl Scorer for LeastLoad {
    fn name(&self) -> &'static str {
        "least-load"
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let (sum, n) = kept
            .iter()
            .filter_map(|&i| candidates[i as usize].load)
            .filter(|r| r.live_rate > 0)
            .fold((0.0f32, 0u32), |(s, n), r| (s + r.live_rate as f32, n + 1));
        let nominal = if n > 0 && sum.is_finite() {
            sum / n as f32
        } else {
            self.default_throughput
        };
        // One short lock: the tally is read, the dispatches a report reflects
        // are released, and each reporting row's credit work is written; the
        // divisions and the KV barrier run after the lock is gone. A row
        // without a report is marked and costed after the peer baseline is
        // known. No allocation.
        let mut reporting = 0usize;
        {
            let mut tally = self.tally.lock();
            for (slot, &i) in out.iter_mut().zip(kept) {
                let c = &candidates[i as usize];
                *slot = match c.load {
                    Some(_) => {
                        let (tokens, requests) = match tally.get_mut(&(c.id, c.dp_rank)) {
                            Some(d) => {
                                d.release_through(c.load_sampled_at_micros);
                                (d.tokens, d.requests)
                            }
                            None => (0, 0),
                        };
                        reporting += 1;
                        self.credit_work(c, tokens, requests)
                    }
                    None => PENDING,
                };
            }
        }
        for (slot, &i) in out.iter_mut().zip(kept) {
            if let Some(r) = candidates[i as usize].load {
                *slot = self.reported_cost(r, *slot, nominal);
            }
        }
        let fleet_has_loads = reporting > 0;
        let mut peer_baseline = 0.0;
        if reporting < kept.len() && fleet_has_loads {
            peer_baseline = out
                .iter()
                .copied()
                .filter(|cost| *cost != PENDING)
                .fold(f32::INFINITY, f32::min);
            if !peer_baseline.is_finite() {
                peer_baseline = 0.0;
            }
        }
        for (slot, &i) in out.iter_mut().zip(kept) {
            if *slot == PENDING {
                *slot = self.unreported_cost(
                    &candidates[i as usize],
                    nominal,
                    fleet_has_loads,
                    peer_baseline,
                );
            }
        }
        // The band of a microsecond of expected wait on the normalized scale:
        // exact for one scorer, additive under several.
        let span = normalize_costs(out);
        if span > 0.0 {
            TieBand((TIE_EPSILON_SECS / span).min(1.0))
        } else {
            TieBand::NORMALIZED
        }
    }

    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            ..Needs::default()
        }
    }
}

// ---------------------------------------------------------------- pickers

/// The best score first, ties drawn uniformly from the decision's seed, then
/// the next best as alternates.
#[derive(Debug)]
pub struct Best;

impl Lifecycle for Best {}

/// Push the kept positions in descending score order, ties by draw: per slot
/// the rows within the band of the best remaining score are one tie, drawn
/// from uniformly. One pass keeps the five best scores (equal scores in scan
/// order); while the next best is below the band of the best remaining, that
/// row stands alone in its band and is the slot's pick with no draw spent.
/// From the first band that holds several rows on, a slot costs two passes:
/// the best remaining score is found first, then the rows within the band of
/// it are drawn from. The taken slots live in a fixed array: nothing is
/// allocated.
fn push_ranked(
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
    band: TieBand,
    out: &mut Targets,
) {
    let slots = MAX_TARGETS.min(kept.len());
    let mut best_five: [(usize, f32); MAX_TARGETS + 1] =
        [(usize::MAX, f32::NEG_INFINITY); MAX_TARGETS + 1];
    let mut len = 0usize;
    for (i, &s) in scores.iter().enumerate() {
        if len == best_five.len() && s <= best_five[len - 1].1 {
            continue;
        }
        let mut at = if len < best_five.len() {
            len += 1;
            len - 1
        } else {
            best_five.len() - 1
        };
        while at > 0 && s > best_five[at - 1].1 {
            best_five[at] = best_five[at - 1];
            at -= 1;
        }
        best_five[at] = (i, s);
    }
    let mut draw = Draw::new(facts.seed);
    let mut taken = [usize::MAX; MAX_TARGETS];
    let mut filled = 0usize;
    while filled < slots
        && (filled + 1 >= len || best_five[filled + 1].1 < best_five[filled].1 - band.0)
    {
        let (i, _) = best_five[filled];
        taken[filled] = i;
        out.push(target_of(&candidates[kept[i] as usize]));
        filled += 1;
    }
    for slot in filled..slots {
        let mut top = f32::NEG_INFINITY;
        for (i, &s) in scores.iter().enumerate() {
            if !taken[..slot].contains(&i) && s > top {
                top = s;
            }
        }
        let mut best: Option<usize> = None;
        let mut tied = 0u32;
        for (i, &s) in scores.iter().enumerate() {
            if taken[..slot].contains(&i) || s < top - band.0 {
                continue;
            }
            tied += 1;
            if best.is_none() || draw.below(tied as usize) == 0 {
                best = Some(i);
            }
        }
        let Some(b) = best else {
            break;
        };
        taken[slot] = b;
        out.push(target_of(&candidates[kept[b] as usize]));
    }
}

fn target_of(c: &Candidate) -> Target {
    Target {
        position: c.position,
        id: c.id,
        dp_rank: c.dp_rank,
        pair: None,
    }
}

impl Picker for Best {
    fn name(&self) -> &'static str {
        "best"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        band: TieBand,
        out: &mut Targets,
    ) {
        push_ranked(facts, candidates, kept, scores, band, out);
    }
}

/// The first survivor, as the host lists them.
#[derive(Debug)]
pub struct Passthrough;

impl Lifecycle for Passthrough {}

impl Picker for Passthrough {
    fn name(&self) -> &'static str {
        "passthrough"
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
        for &i in kept.iter().take(MAX_TARGETS) {
            out.push(target_of(&candidates[i as usize]));
        }
    }
}

/// A uniform draw over the survivors from the decision's seed, then further
/// distinct draws as alternates.
#[derive(Debug)]
pub struct Random;

impl Lifecycle for Random {}

impl Picker for Random {
    fn name(&self) -> &'static str {
        "random"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        _: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let mut draw = Draw::new(facts.seed);
        let n = kept.len();
        let mut taken = [usize::MAX; MAX_TARGETS];
        for slot in 0..MAX_TARGETS.min(n) {
            // The j-th position among those not taken yet: a partial
            // Fisher-Yates without the copy of the survivor list.
            let j = draw.below(n - slot);
            let mut seen = 0usize;
            let mut chosen = 0usize;
            for i in 0..n {
                if taken[..slot].contains(&i) {
                    continue;
                }
                if seen == j {
                    chosen = i;
                    break;
                }
                seen += 1;
            }
            taken[slot] = chosen;
            out.push(target_of(&candidates[kept[chosen] as usize]));
        }
    }
}

/// Two uniform draws, the better score taken; the other is the alternate.
/// The profile's scorers decide which of the two is better, so a profile
/// with this picker names at least one scorer.
#[derive(Debug)]
pub struct PowerOfTwo;

impl Lifecycle for PowerOfTwo {}

impl Picker for PowerOfTwo {
    fn name(&self) -> &'static str {
        "power-of-two"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        if kept.is_empty() {
            return;
        }
        let mut draw = Draw::new(facts.seed);
        let a = draw.below(kept.len());
        if kept.len() == 1 {
            out.push(target_of(&candidates[kept[a] as usize]));
            return;
        }
        let mut b = draw.below(kept.len() - 1);
        if b >= a {
            b += 1;
        }
        let (first, second) = if scores[a] >= scores[b] {
            (a, b)
        } else {
            (b, a)
        };
        out.push(target_of(&candidates[kept[first] as usize]));
        out.push(target_of(&candidates[kept[second] as usize]));
    }
}

/// `a` before `b` in the count-first order: fewer requests in flight on
/// the host, then the higher score (compared exactly, as the policies'
/// expected-wait tie-break is), equal on both.
fn count_first_order(
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
    a: usize,
    b: usize,
) -> std::cmp::Ordering {
    let count_a = candidates[kept[a] as usize].in_flight;
    let count_b = candidates[kept[b] as usize].in_flight;
    count_a
        .cmp(&count_b)
        .then_with(|| scores[b].total_cmp(&scores[a]))
}

/// The fewest requests in flight on the host first, the profile's score
/// among equal counts, the decision's seed among exact ties; the next by
/// the same order as alternates. No weight or band stands between the two
/// stages: a strictly lower count wins whatever the scores.
#[derive(Debug)]
pub struct CountFirst;

impl Lifecycle for CountFirst {}

impl Picker for CountFirst {
    fn name(&self) -> &'static str {
        "count-first"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        let mut draw = Draw::new(facts.seed);
        let mut taken = [usize::MAX; MAX_TARGETS];
        for slot in 0..MAX_TARGETS.min(kept.len()) {
            let mut best: Option<usize> = None;
            let mut tied = 0u32;
            for i in 0..kept.len() {
                if taken[..slot].contains(&i) {
                    continue;
                }
                let Some(b) = best else {
                    best = Some(i);
                    tied = 1;
                    continue;
                };
                match count_first_order(candidates, kept, scores, i, b) {
                    std::cmp::Ordering::Less => {
                        best = Some(i);
                        tied = 1;
                    }
                    std::cmp::Ordering::Equal => {
                        tied += 1;
                        if draw.below(tied as usize) == 0 {
                            best = Some(i);
                        }
                    }
                    std::cmp::Ordering::Greater => {}
                }
            }
            let Some(b) = best else {
                break;
            };
            taken[slot] = b;
            out.push(target_of(&candidates[kept[b] as usize]));
        }
    }
}

/// Two uniform draws, the one with fewer requests in flight taken and the
/// profile's score deciding equal counts; the other is the alternate.
#[derive(Debug)]
pub struct PowerOfTwoCountFirst;

impl Lifecycle for PowerOfTwoCountFirst {}

impl Picker for PowerOfTwoCountFirst {
    fn name(&self) -> &'static str {
        "power-of-two-count-first"
    }

    fn pick(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        scores: &[f32],
        _: TieBand,
        out: &mut Targets,
    ) {
        if kept.is_empty() {
            return;
        }
        let mut draw = Draw::new(facts.seed);
        let a = draw.below(kept.len());
        if kept.len() == 1 {
            out.push(target_of(&candidates[kept[a] as usize]));
            return;
        }
        let mut b = draw.below(kept.len() - 1);
        if b >= a {
            b += 1;
        }
        let (first, second) =
            if count_first_order(candidates, kept, scores, b, a) == std::cmp::Ordering::Less {
                (b, a)
            } else {
                (a, b)
            };
        out.push(target_of(&candidates[kept[first] as usize]));
        out.push(target_of(&candidates[kept[second] as usize]));
    }
}

/// A rotation per survivor set, keyed by the set's rows (id and DP rank) in
/// order; a set seen first starts from a shared advancing position. The table
/// holds 4096 sets and is cleared when full, so no decision walks
/// it: a set seen again after a clear starts from the shared position.
#[derive(Debug)]
pub struct RoundRobin {
    sets: Mutex<RotationTable>,
}

#[derive(Debug, Default)]
struct RotationTable {
    rotations: HashMap<u64, u64>,
    next_start: u64,
}

/// Distinct survivor sets one instance keeps a rotation for.
const MAX_TRACKED_SETS: usize = 4096;

impl Default for RoundRobin {
    fn default() -> Self {
        Self::new()
    }
}

impl RoundRobin {
    pub fn new() -> Self {
        Self {
            sets: Mutex::new(RotationTable::default()),
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

    fn advance(&self, key: u64) -> u64 {
        let mut table = self.sets.lock();
        if let Some(next) = table.rotations.get_mut(&key) {
            let position = *next;
            *next += 1;
            return position;
        }
        if table.rotations.len() >= MAX_TRACKED_SETS {
            table.rotations.clear();
        }
        let start = table.next_start;
        table.next_start += 1;
        table.rotations.insert(key, start + 1);
        start
    }
}

impl Lifecycle for RoundRobin {}

impl Picker for RoundRobin {
    fn name(&self) -> &'static str {
        "round-robin"
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
    use crate::view::{Health, LoadRow};

    fn candidate(position: u32, in_flight: u32) -> Candidate {
        Candidate {
            position,
            id: WorkerId(1_000 + u64::from(position)),
            health: Health {
                healthy: true,
                ..Health::default()
            },
            in_flight,
            ..Candidate::default()
        }
    }

    fn kept(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

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

    #[test]
    fn least_requests_ranks_by_in_flight_and_best_orders_alternates() {
        let pool = [candidate(0, 4), candidate(1, 0), candidate(2, 2)];
        let mut scores = [0.0f32; 3];
        let band = LeastRequests.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 1.0, 0.5]);
        assert_eq!(band, TieBand::NORMALIZED);
        let mut out = Targets::new();
        Best.pick(
            &RequestFacts::default(),
            &pool,
            &kept(3),
            &scores,
            band,
            &mut out,
        );
        let order: Vec<u32> = out.iter().map(|t| t.position).collect();
        assert_eq!(order, vec![1, 2, 0]);
    }

    #[test]
    fn equal_scores_spread_across_seeds() {
        let pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        let mut picks = [0usize; 3];
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(
                &facts,
                &pool,
                &kept(3),
                &[1.0; 3],
                TieBand::NORMALIZED,
                &mut out,
            );
            picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        assert!(picks.iter().all(|&n| n > 50), "{picks:?}");
    }

    #[test]
    fn random_draws_distinct_alternates_and_spreads_across_seeds() {
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let mut picks = [0usize; 3];
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Random.pick(
                &facts,
                &pool,
                &kept(3),
                &[0.0; 3],
                TieBand::NORMALIZED,
                &mut out,
            );
            let positions: Vec<u32> = out.iter().map(|t| t.position).collect();
            let mut distinct = positions.clone();
            distinct.sort_unstable();
            distinct.dedup();
            assert_eq!(distinct.len(), 3, "{positions:?}");
            picks[positions[0] as usize] += 1;
        }
        assert!(picks.iter().all(|&n| n > 50), "{picks:?}");
    }

    fn report(running: u32, waiting_uncached_tokens: u32) -> LoadRow {
        LoadRow {
            running,
            waiting_uncached_tokens,
            live_rate: 2000,
            ..LoadRow::default()
        }
    }

    #[test]
    fn least_load_prices_work_over_drain_and_credits_a_dispatch_until_its_report() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(report(1, 0));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(report(0, 4096));
        pool[1].load_sampled_at_micros = 1_000;
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            scores,
            [1.0, 0.0],
            "one running request beats a queue of 4096 tokens"
        );

        // A dispatch after the report's sample stays as credit...
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 8_192,
            ..Lease::default()
        };
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(ll.held(pool[0].id), 1);
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0], "the credited worker now costs more");

        // ...and a report sampled after the dispatch releases it.
        pool[0].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(ll.held(pool[0].id), 0);
        assert_eq!(scores, [1.0, 0.0]);

        // A removed worker leaves no tally behind.
        ll.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        ll.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(ll.held(pool[1].id), 0);
    }

    #[test]
    fn a_credit_dates_from_the_dispatch_and_not_from_the_decision() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(report(0, 0));
        let lease = Lease {
            issued_micros: 100,
            tokens: 10,
            ..Lease::default()
        };
        // Decided at 100, held in a queue, dispatched at 300.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 200);
        let mut scores = [0.0f32; 1];
        pool[0].load_sampled_at_micros = 200;
        ll.score(&RequestFacts::default(), &pool, &kept(1), &mut scores);
        assert_eq!(
            ll.held(pool[0].id),
            1,
            "a sample before the dispatch keeps the credit"
        );
        pool[0].load_sampled_at_micros = 300;
        ll.score(&RequestFacts::default(), &pool, &kept(1), &mut scores);
        assert_eq!(
            ll.held(pool[0].id),
            0,
            "a sample at the dispatch releases it"
        );
    }

    #[test]
    fn a_membership_snapshot_keeps_only_the_tallies_of_its_ids() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool = [candidate(0, 0), candidate(1, 0)];
        let lease = Lease::default();
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        ll.on_membership(&Membership::Snapshot(&[pool[0].id]));
        assert_eq!(ll.held(pool[0].id), 1);
        assert_eq!(
            ll.held(pool[1].id),
            0,
            "a worker the snapshot lacks is gone"
        );
    }

    #[test]
    fn a_dark_fleet_scores_by_in_flight_and_a_missing_report_by_the_peer_baseline() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool = [candidate(0, 3), candidate(1, 1)];
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        let mut mixed = pool;
        mixed[0].load = Some(report(0, 0));
        mixed[0].in_flight = 0;
        ll.score(&RequestFacts::default(), &mixed, &kept(2), &mut scores);
        assert_eq!(scores[0], 1.0, "the idle reporting worker wins");
        assert!(
            scores[1] < 1.0,
            "the dark worker starts from the baseline plus its in-flight"
        );
    }

    #[test]
    fn kv_pressure_raises_the_cost_of_an_otherwise_equal_worker() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(report(1, 0));
        pool[1].load = Some(LoadRow {
            kv_usage_permille: 900,
            active_kv_permille: 900,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [1.0, 0.0]);
    }

    /// A full but reclaimable prefix cache is no pressure: the idle worker
    /// whose running requests hold none of its cache beats the busy one
    /// holding half, whatever the raw usage says (the legacy policy's case).
    #[test]
    fn reclaimable_prefix_cache_does_not_make_an_idle_worker_busy() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(LoadRow {
            kv_usage_permille: 1000,
            active_kv_permille: 0,
            ..report(0, 0)
        });
        pool[1].load = Some(LoadRow {
            kv_usage_permille: 500,
            active_kv_permille: 500,
            ..report(1, 0)
        });
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores[0], 1.0, "the idle worker wins");
        assert!(
            scores[1] < 1.0,
            "the busy worker's held share is the pressure"
        );
    }

    #[test]
    fn least_load_ties_within_a_microsecond_of_wait_on_a_spread_pool() {
        // Two workers a quarter of a microsecond apart in expected wait, and a
        // third four seconds behind that stretches the normalized scale: the
        // two still tie, and a pair two microseconds apart still does not.
        // (The band is exact down to the normalized scale's resolution: a
        // difference under about 6e-8 of the spread is one f32 value, so a
        // spread past about 16 s widens a tie beyond the microsecond.)
        let ll = LeastLoad::new(0.0, 1024, 1_000_000.0);
        let near = |tokens: u32| LoadRow {
            waiting_uncached_tokens: tokens,
            live_rate: 4_000_000,
            ..LoadRow::default()
        };
        let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        pool[0].load = Some(near(4_000_000)); // 1.0 s
        pool[1].load = Some(near(4_000_001)); // 1.0 s + 0.25 us
        pool[2].load = Some(near(20_000_000)); // 5 s
        let mut scores = [0.0f32; 3];
        let band = ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert!(
            band.0 < TieBand::NORMALIZED.0,
            "a spread pool narrows the band: {band:?}"
        );
        let mut picks = [0usize; 3];
        for seed in 0..200u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        assert!(picks[0] > 40 && picks[1] > 40 && picks[2] == 0, "{picks:?}");

        pool[1].load = Some(near(4_000_008)); // 1.0 s + 2 us
        let band = ll.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        let mut out = Targets::new();
        for seed in 0..50u64 {
            out.clear();
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            assert_eq!(
                out.first().map(|t| t.position),
                Some(0),
                "two microseconds apart is no tie"
            );
        }
    }

    #[test]
    fn a_cold_fleet_spreads_its_picks_across_every_worker() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool: Vec<Candidate> = (0..128)
            .map(|i| {
                let mut c = candidate(i, 0);
                c.load = Some(report(0, 0));
                c
            })
            .collect();
        let kept = kept(128);
        let mut scores = vec![0.0f32; 128];
        let band = ll.score(&RequestFacts::default(), &pool, &kept, &mut scores);
        let mut hits = vec![0usize; 128];
        for seed in 0..1_000u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept, &scores, band, &mut out);
            hits[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        let idle = hits.iter().filter(|&&n| n == 0).count();
        assert!(
            idle <= 2,
            "{idle} of 128 cold workers never picked in a thousand draws"
        );
        assert!(
            hits.iter().all(|&n| n < 40),
            "a cold worker took {} of a thousand",
            hits.iter().max().unwrap_or(&0)
        );
    }

    #[test]
    fn power_of_two_takes_the_better_scored_of_two_draws() {
        let pool = [candidate(0, 9), candidate(1, 9), candidate(2, 0)];
        let mut scores = [0.0f32; 3];
        LeastRequests.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        let mut hits = 0;
        for seed in 0..100u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            PowerOfTwo.pick(
                &facts,
                &pool,
                &kept(3),
                &scores,
                TieBand::NORMALIZED,
                &mut out,
            );
            assert_eq!(out.len(), 2);
            if out.first().map(|t| t.position) == Some(2) {
                hits += 1;
            }
        }
        // Worker 2 is drawn in two of three pairs and wins each time.
        assert!((50..=85).contains(&hits), "{hits}");
    }

    /// The review's case: counts 0, 1 and 2000 with the idle worker's wait
    /// the worst of the three. The count decides before any score: the idle
    /// worker leads, the next by count follows; a drawn pair is led by its
    /// fewer in flight, so the busiest worker never leads one.
    #[test]
    fn count_first_takes_the_fewest_in_flight_before_any_score() {
        let pool = [candidate(0, 0), candidate(1, 1), candidate(2, 2000)];
        let scores = [0.0f32, 1.0, 0.49];
        for seed in 0..20u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            CountFirst.pick(
                &facts,
                &pool,
                &kept(3),
                &scores,
                TieBand::NORMALIZED,
                &mut out,
            );
            let order: Vec<u32> = out.iter().map(|t| t.position).collect();
            assert_eq!(order, [0, 1, 2], "seed {seed}");
            let mut pair = Targets::new();
            PowerOfTwoCountFirst.pick(
                &facts,
                &pool,
                &kept(3),
                &scores,
                TieBand::NORMALIZED,
                &mut pair,
            );
            let drawn: Vec<u32> = pair.iter().map(|t| t.position).collect();
            assert_eq!(drawn.len(), 2, "seed {seed}");
            assert!(
                pool[drawn[0] as usize].in_flight <= pool[drawn[1] as usize].in_flight,
                "seed {seed}: {drawn:?}"
            );
            assert_ne!(drawn[0], 2, "seed {seed}: the busiest worker never leads");
        }
    }

    /// Equal counts fall to the score, compared exactly; an exact tie on
    /// both is drawn from the seed; the alternates keep the order.
    #[test]
    fn count_first_breaks_equal_counts_by_the_score_and_exact_ties_by_the_seed() {
        let pool = [
            candidate(0, 3),
            candidate(1, 3),
            candidate(2, 3),
            candidate(3, 7),
        ];
        let scores = [0.2f32, 0.9, 0.9, 1.0];
        let mut firsts = std::collections::BTreeSet::new();
        for seed in 0..40u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            CountFirst.pick(
                &facts,
                &pool,
                &kept(4),
                &scores,
                TieBand::NORMALIZED,
                &mut out,
            );
            let order: Vec<u32> = out.iter().map(|t| t.position).collect();
            assert!(order[0] == 1 || order[0] == 2, "{order:?}");
            assert_eq!((order[2], order[3]), (0, 3), "{order:?}");
            firsts.insert(order[0]);
        }
        assert_eq!(firsts.len(), 2, "an exact tie is drawn from the seed");
    }

    #[test]
    fn passthrough_keeps_the_hosts_order() {
        let pool = [candidate(0, 9), candidate(1, 0)];
        let mut out = Targets::new();
        Passthrough.pick(
            &RequestFacts::default(),
            &pool,
            &[1, 0],
            &[0.0; 2],
            TieBand::NORMALIZED,
            &mut out,
        );
        let order: Vec<u32> = out.iter().map(|t| t.position).collect();
        assert_eq!(order, vec![1, 0]);
    }

    #[test]
    fn a_dispatch_credits_the_worker_of_its_side() {
        let scorer = LeastLoad::new(0.15, 1024, 2000.0);
        let lease = Lease {
            issued_micros: 10,
            tokens: 100,
            ..Lease::default()
        };
        let target = Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(crate::decision::Pair {
                decode_position: 0,
                decode_id: WorkerId(2),
                decode_dp_rank: 0,
            }),
        };
        scorer.on_dispatched(&lease, &target, Side::Prefill, 0);
        scorer.on_dispatched(&lease, &target, Side::Decode, 0);
        assert_eq!(
            scorer.held(WorkerId(1)),
            1,
            "the prefill side credits the prefill"
        );
        assert_eq!(
            scorer.held(WorkerId(2)),
            1,
            "the decode side credits the pair's decode"
        );
        let single = Target {
            pair: None,
            ..target
        };
        scorer.on_dispatched(&lease, &single, Side::Single, 0);
        assert_eq!(scorer.held(WorkerId(1)), 2);
    }

    #[test]
    fn two_ranks_of_one_worker_are_credited_and_released_apart() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        for (rank, c) in pool.iter_mut().enumerate() {
            c.id = WorkerId(7);
            c.dp_rank = rank as u16;
            c.dp_size = 2;
            c.load = Some(report(0, 0));
            c.load_sampled_at_micros = 1_000;
        }
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 8_192,
            ..Lease::default()
        };
        // A dispatch to rank 0 is rank 0's credit alone.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(
            (ll.held_at(WorkerId(7), 0), ll.held_at(WorkerId(7), 1)),
            (1, 0)
        );
        assert_eq!(ll.held(WorkerId(7)), 1);
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            scores,
            [0.0, 1.0],
            "the credited rank costs more than its idle sibling"
        );
        // Rank 1's later report releases nothing of rank 0's.
        pool[1].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(
            ll.held_at(WorkerId(7), 0),
            1,
            "a sibling rank's report is not this rank's"
        );
        assert_eq!(scores, [0.0, 1.0]);
        // Rank 0's own report sampled after the dispatch does.
        pool[0].load_sampled_at_micros = 3_000;
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(ll.held(WorkerId(7)), 0);
        assert_eq!(scores, [1.0, 1.0]);
        // The decode side of a pair credits the decode's own rank.
        let pair = Target {
            position: 0,
            id: WorkerId(1),
            dp_rank: 0,
            pair: Some(crate::decision::Pair {
                decode_position: 1,
                decode_id: WorkerId(7),
                decode_dp_rank: 1,
            }),
        };
        ll.on_dispatched(&lease, &pair, Side::Decode, 0);
        assert_eq!(
            (ll.held_at(WorkerId(7), 0), ll.held_at(WorkerId(7), 1)),
            (0, 1)
        );
        // A resize starts every rank of the worker over; a removal takes them.
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_membership(&Membership::Resized(WorkerId(7)));
        assert_eq!(ll.held(WorkerId(7)), 0);
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        ll.on_dispatched(&lease, &pair, Side::Decode, 0);
        ll.on_membership(&Membership::Removed(WorkerId(7)));
        assert_eq!(ll.held(WorkerId(7)), 0);
    }

    #[test]
    fn single_bands_are_taken_in_order_and_a_shared_band_behind_them_is_drawn_from() {
        // Two rows alone in their bands lead every pick in order; the three
        // rows that share the next band take the third slot by the seed, and
        // the fourth slot is another of them.
        let pool = [
            candidate(0, 0),
            candidate(1, 0),
            candidate(2, 0),
            candidate(3, 0),
            candidate(4, 0),
        ];
        let scores = [0.9f32, 1.0, 0.5, 0.5, 0.5];
        let mut thirds = [0u32; 5];
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(
                &facts,
                &pool,
                &kept(5),
                &scores,
                TieBand::NORMALIZED,
                &mut out,
            );
            let order: Vec<u32> = out.iter().map(|t| t.position).collect();
            assert_eq!(&order[..2], &[1, 0], "seed {seed}: {order:?}");
            assert_eq!(order.len(), 4, "seed {seed}");
            assert!(
                order[2] >= 2 && order[3] >= 2 && order[2] != order[3],
                "seed {seed}: {order:?}"
            );
            thirds[order[2] as usize] += 1;
        }
        assert!(
            thirds[2] > 50 && thirds[3] > 50 && thirds[4] > 50,
            "the shared band is drawn from: {thirds:?}"
        );
    }

    #[test]
    fn the_four_best_of_sixty_four_rows_are_picked_in_order_wherever_they_sit() {
        // The five-best array fills on the first rows and shifts for every
        // better one after it; the four best must come out in order whether
        // they sit scattered in a rising pool or arrive last, rising.
        let pool: Vec<Candidate> = (0..64).map(|i| candidate(i, 0)).collect();
        let kept = kept(64);
        let mut scattered: Vec<f32> = (0..64).map(|i| 0.1 + 0.004 * i as f32).collect();
        for (position, score) in [(7usize, 1.0f32), (40, 0.99), (63, 0.98), (22, 0.97)] {
            scattered[position] = score;
        }
        let mut last_rising: Vec<f32> = (0..64).map(|i| 0.1 + 0.004 * i as f32).collect();
        for (position, score) in [(60usize, 0.96f32), (61, 0.97), (62, 0.98), (63, 0.99)] {
            last_rising[position] = score;
        }
        for (scores, expected) in [
            (scattered, [7u32, 40, 63, 22]),
            (last_rising, [63, 62, 61, 60]),
        ] {
            for seed in 0..8u64 {
                let facts = RequestFacts {
                    seed,
                    ..RequestFacts::default()
                };
                let mut out = Targets::new();
                Best.pick(&facts, &pool, &kept, &scores, TieBand::NORMALIZED, &mut out);
                let order: Vec<u32> = out.iter().map(|t| t.position).collect();
                assert_eq!(order, expected, "seed {seed}");
            }
        }
    }

    #[test]
    fn a_wide_band_ties_against_the_best_score_and_not_the_running_best() {
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let scores = [0.95f32, 1.0, 0.92];
        let mut firsts = [0u32; 3];
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &[0, 1, 2], &scores, TieBand(0.06), &mut out);
            firsts[out.first().expect("a pick").position as usize] += 1;
        }
        assert_eq!(
            firsts[2], 0,
            "0.92 is outside the band of the best score 1.0"
        );
        assert!(
            firsts[0] > 100 && firsts[1] > 100,
            "the two tied rows share the draws: {firsts:?}"
        );
    }

    #[test]
    fn the_filters_read_their_bits() {
        let facts = RequestFacts::default();
        let cx = Constraints::default();
        let mut c = candidate(0, 0);
        assert!(Healthy.keep(&facts, &c, &cx));
        for down in [
            Health {
                healthy: false,
                ..c.health
            },
            Health {
                breaker_open: true,
                ..c.health
            },
            Health {
                stalled: true,
                ..c.health
            },
            Health {
                draining: true,
                ..c.health
            },
        ] {
            let unroutable = Candidate { health: down, ..c };
            assert!(!Healthy.keep(&facts, &unroutable, &cx));
        }
        assert!(NotOverloaded.keep(&facts, &c, &cx));
        c.health.overloaded = true;
        assert!(!NotOverloaded.keep(&facts, &c, &cx));
        let lifted = Constraints {
            overload_lifted: true,
            ..cx
        };
        assert!(NotOverloaded.keep(&facts, &c, &lifted));
        assert!(!HasCapacity.keep(&facts, &c, &cx));
        c.has_capacity = true;
        assert!(HasCapacity.keep(&facts, &c, &cx));
        assert!(!HasRoom.keep(&facts, &c, &cx));
        c.has_room = true;
        assert!(HasRoom.keep(&facts, &c, &cx));
        let tried = [c.id];
        let retry = RequestFacts {
            tried: &tried,
            ..facts
        };
        assert!(!NotTried.keep(&retry, &c, &cx));
        let excluded = Constraints {
            excluded: &tried,
            ..cx
        };
        assert!(!NotTried.keep(&facts, &c, &excluded));
        assert!(NotTried.soft() && HasRoom.soft() && !Healthy.soft() && !HasCapacity.soft());
    }

    #[test]
    fn the_affinity_scorers_read_the_rows_facts() {
        let hashes = [1u64, 2, 3, 4];
        let facts = RequestFacts {
            block_hashes: &hashes,
            ..RequestFacts::default()
        };
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].overlap_blocks = 2;
        pool[1].session_hit = true;
        let mut scores = [0.0f32; 2];
        PrefixAffinity.score(&facts, &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.5, 0.0]);
        SessionAffinity.score(&facts, &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        assert!(PrefixAffinity.needs().token_hashes && PrefixAffinity.needs().events);
        assert!(SessionAffinity.needs().headers);
        assert!(LeastLoad::new(0.15, 1024, 2000.0).needs().loads);
    }
}
