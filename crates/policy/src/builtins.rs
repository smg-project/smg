//! The built-in stages: filters, scorers and pickers the presets compose.
//! Every stateful one keeps its state behind the lifecycle alone.

use std::collections::{HashMap, VecDeque};

use parking_lot::Mutex;

use crate::{
    decision::{Target, Targets, MAX_TARGETS},
    framework::{Filter, Picker, Scorer},
    lease::{Lease, Lifecycle, Membership, Side},
    spec::Needs,
    view::{Candidate, Constraints, RequestFacts, WorkerId},
    Draw,
};

/// Scores within this of the best are one tie, drawn from uniformly.
const TIE_EPSILON: f32 = 1e-6;

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
fn normalize_costs(out: &mut [f32]) {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    for &c in out.iter() {
        lo = lo.min(c);
        hi = hi.max(c);
    }
    let span = hi - lo;
    for o in out.iter_mut() {
        *o = if span > 0.0 && span.is_finite() {
            1.0 - (*o - lo) / span
        } else {
            1.0
        };
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

    fn score(&self, _: &RequestFacts<'_>, candidates: &[Candidate], kept: &[u32], out: &mut [f32]) {
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = candidates[i as usize].in_flight as f32;
        }
        normalize_costs(out);
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

    fn score(&self, _: &RequestFacts<'_>, candidates: &[Candidate], kept: &[u32], out: &mut [f32]) {
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = if candidates[i as usize].session_hit {
                1.0
            } else {
                0.0
            };
        }
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
    ) {
        let blocks = facts.block_hashes.len().max(1) as f32;
        for (o, &i) in out.iter_mut().zip(kept) {
            *o = (candidates[i as usize].overlap_blocks as f32 / blocks).clamp(0.0, 1.0);
        }
    }

    fn needs(&self) -> Needs {
        Needs {
            token_hashes: true,
            events: true,
            ..Needs::default()
        }
    }
}

/// Dispatches to one worker no report reflects yet, oldest first.
#[derive(Debug, Default)]
struct SinceReport {
    tokens: u64,
    requests: u64,
    dispatches: VecDeque<(u64, u32)>,
}

/// Dispatches kept per worker between reports.
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

/// The expected wait: token-work over drain rate plus a convex KV-pressure
/// barrier, the since-report dispatch credit water-filling arrivals between
/// reports. The host's in-flight count floors the report.
#[derive(Debug)]
pub struct LeastLoad {
    pub kv_pressure_weight: f32,
    pub mean_prefill_tokens: u32,
    pub default_throughput: f32,
    tally: Mutex<HashMap<WorkerId, SinceReport>>,
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
            tally: Mutex::new(HashMap::new()),
        }
    }

    /// Dispatches no report reflects yet on `worker`, for tests and dumps.
    pub fn held(&self, worker: WorkerId) -> usize {
        self.tally
            .lock()
            .get(&worker)
            .map_or(0, |d| d.dispatches.len())
    }

    fn cost(
        &self,
        c: &Candidate,
        credit_tokens: u64,
        credit_requests: u64,
        nominal: f32,
        fleet_has_loads: bool,
        peer_baseline: f32,
    ) -> f32 {
        let mean = self.mean_prefill_tokens as f32;
        match c.report {
            Some(r) => {
                let queued = if r.waiting_uncached_tokens > 0 {
                    r.waiting_uncached_tokens as f32
                } else {
                    r.waiting as f32 * mean
                };
                let running = r.running as f32 * mean;
                let reported = u64::from(r.running) + u64::from(r.waiting);
                let unaccounted = u64::from(c.in_flight).saturating_sub(reported + credit_requests);
                let work = queued + running + unaccounted as f32 * mean + credit_tokens as f32;
                let drain = if r.live_rate > 0.0 {
                    r.live_rate
                } else {
                    nominal
                };
                let k = r.kv_pressure.clamp(0.0, 0.999_999);
                work / drain + self.kv_pressure_weight * k / (1.0 - k)
            }
            None if fleet_has_loads => peer_baseline + c.in_flight as f32 * mean / nominal,
            None => c.in_flight as f32,
        }
    }
}

/// A cost slot not written yet: a candidate without a report is costed
/// after the reporting peers' baseline is known. Costs are never negative.
const PENDING: f32 = f32::NEG_INFINITY;

impl Lifecycle for LeastLoad {
    fn on_dispatched(&self, lease: &Lease, target: &Target, _side: Side) {
        let mut tally = self.tally.lock();
        tally
            .entry(target.id)
            .or_default()
            .record(lease.issued_micros, lease.tokens);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        match change {
            Membership::Removed(id) | Membership::Replaced { old: id, .. } => {
                self.tally.lock().remove(id);
            }
            _ => {}
        }
    }
}

impl Scorer for LeastLoad {
    fn name(&self) -> &'static str {
        "least-load"
    }

    fn score(&self, _: &RequestFacts<'_>, candidates: &[Candidate], kept: &[u32], out: &mut [f32]) {
        let (sum, n) = kept
            .iter()
            .filter_map(|&i| candidates[i as usize].report)
            .filter(|r| r.live_rate > 0.0)
            .fold((0.0f32, 0u32), |(s, n), r| (s + r.live_rate, n + 1));
        let nominal = if n > 0 && sum.is_finite() {
            sum / n as f32
        } else {
            self.default_throughput
        };
        // One short lock: the reporting candidates' costs are written as the
        // credits stand after each report's sample (the dispatches a report
        // reflects are released here too); a candidate without a report is
        // marked and costed after the peer baseline is known. No allocation.
        let mut reporting = 0usize;
        {
            let mut tally = self.tally.lock();
            for (slot, &i) in out.iter_mut().zip(kept) {
                let c = &candidates[i as usize];
                *slot = match c.report {
                    Some(r) => {
                        let (tokens, requests) = match tally.get_mut(&c.id) {
                            Some(d) => {
                                d.release_through(r.sampled_at_micros);
                                (d.tokens, d.requests)
                            }
                            None => (0, 0),
                        };
                        reporting += 1;
                        self.cost(c, tokens, requests, nominal, true, 0.0)
                    }
                    None => PENDING,
                };
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
                *slot = self.cost(
                    &candidates[i as usize],
                    0,
                    0,
                    nominal,
                    fleet_has_loads,
                    peer_baseline,
                );
            }
        }
        normalize_costs(out);
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

/// Push the kept positions in descending score order, ties by draw.
fn push_ranked(
    facts: &RequestFacts<'_>,
    candidates: &[Candidate],
    kept: &[u32],
    scores: &[f32],
    out: &mut Targets,
) {
    let mut draw = Draw::new(facts.seed);
    let mut taken = vec![false; kept.len()];
    for _ in 0..MAX_TARGETS.min(kept.len()) {
        let mut best: Option<usize> = None;
        let mut tied = 0u32;
        for (i, &s) in scores.iter().enumerate() {
            if taken[i] {
                continue;
            }
            match best {
                None => {
                    best = Some(i);
                    tied = 1;
                }
                Some(b) if s > scores[b] + TIE_EPSILON => {
                    best = Some(i);
                    tied = 1;
                }
                Some(b) if (s - scores[b]).abs() <= TIE_EPSILON => {
                    tied += 1;
                    if draw.below(tied as usize) == 0 {
                        best = Some(i);
                    }
                }
                Some(_) => {}
            }
        }
        let Some(b) = best else {
            break;
        };
        taken[b] = true;
        out.push(target_of(&candidates[kept[b] as usize]));
    }
}

fn target_of(c: &Candidate) -> Target {
    Target {
        position: c.position,
        id: c.id,
        dp_rank: c.dp_rank,
        decode: None,
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
        out: &mut Targets,
    ) {
        push_ranked(facts, candidates, kept, scores, out);
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
        out: &mut Targets,
    ) {
        let mut draw = Draw::new(facts.seed);
        let mut order: Vec<u32> = kept.to_vec();
        // A partial Fisher-Yates over the first MAX_TARGETS positions.
        for i in 0..MAX_TARGETS.min(order.len()) {
            let j = i + draw.below(order.len() - i);
            order.swap(i, j);
            out.push(target_of(&candidates[order[i] as usize]));
        }
    }
}

/// Two uniform draws, the less loaded taken; the other is the alternate.
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
        // With scorers the profile's combined score decides; without, the
        // host's in-flight count does.
        let better = |x: usize, y: usize| {
            if scores.iter().any(|&s| s != scores[0]) {
                scores[x] >= scores[y]
            } else {
                candidates[kept[x] as usize].in_flight <= candidates[kept[y] as usize].in_flight
            }
        };
        let (first, second) = if better(a, b) { (a, b) } else { (b, a) };
        out.push(target_of(&candidates[kept[first] as usize]));
        out.push(target_of(&candidates[kept[second] as usize]));
    }
}

/// A rotation per survivor set, keyed by the set's ids in order, bounded by
/// an LRU of sets; a set seen first starts from a shared advancing position.
#[derive(Debug)]
pub struct RoundRobin {
    sets: Mutex<RotationTable>,
}

#[derive(Debug, Default)]
struct RotationTable {
    rotations: HashMap<u64, (u64, u64)>,
    next_start: u64,
    tick: u64,
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

    fn set_key(candidates: &[Candidate], kept: &[u32]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &i in kept {
            h ^= candidates[i as usize].id.0;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
        h
    }

    fn advance(&self, key: u64) -> u64 {
        let mut table = self.sets.lock();
        table.tick += 1;
        let tick = table.tick;
        if let Some((next, last_used)) = table.rotations.get_mut(&key) {
            *last_used = tick;
            let position = *next;
            *next += 1;
            return position;
        }
        if table.rotations.len() >= MAX_TRACKED_SETS {
            if let Some(stale) = table
                .rotations
                .iter()
                .min_by_key(|(_, &(_, last_used))| last_used)
                .map(|(&k, _)| k)
            {
                table.rotations.remove(&stale);
            }
        }
        let start = table.next_start;
        table.next_start += 1;
        table.rotations.insert(key, (start + 1, tick));
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
    use crate::view::{Health, LoadReport};

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

    #[test]
    fn round_robin_walks_a_set_in_turn_and_keeps_sets_apart() {
        let rr = RoundRobin::new();
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let facts = RequestFacts::default();
        let mut seen = Vec::new();
        for _ in 0..6 {
            let mut out = Targets::new();
            rr.pick(&facts, &pool, &kept(3), &[0.0; 3], &mut out);
            seen.push(out.first().map(|t| t.position).unwrap_or(u32::MAX));
        }
        let start = seen[0];
        for (k, &p) in seen.iter().enumerate() {
            assert_eq!(p, (start as usize + k) as u32 % 3, "{seen:?}");
        }
        // Another set has its own rotation.
        let mut out = Targets::new();
        rr.pick(&facts, &pool, &[0, 1], &[0.0; 2], &mut out);
        assert!(out.first().is_some());
    }

    #[test]
    fn least_requests_ranks_by_in_flight_and_best_orders_alternates() {
        let pool = [candidate(0, 4), candidate(1, 0), candidate(2, 2)];
        let mut scores = [0.0f32; 3];
        LeastRequests.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 1.0, 0.5]);
        let mut out = Targets::new();
        Best.pick(&RequestFacts::default(), &pool, &kept(3), &scores, &mut out);
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
            Best.pick(&facts, &pool, &kept(3), &[1.0; 3], &mut out);
            picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        assert!(picks.iter().all(|&n| n > 50), "{picks:?}");
    }

    #[test]
    fn least_load_prices_work_over_drain_and_credits_a_dispatch_until_its_report() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let report = |running: u32, waiting_uncached_tokens: u32| LoadReport {
            running,
            waiting_uncached_tokens,
            live_rate: 2000.0,
            sampled_at_micros: 1_000,
            ..LoadReport::default()
        };
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].report = Some(report(1, 0));
        pool[1].report = Some(report(0, 4096));
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
        ll.on_dispatched(&lease, &target_of(&pool[0]), Side::Single);
        assert_eq!(ll.held(pool[0].id), 1);
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0], "the credited worker now costs more");

        // ...and a report sampled after the dispatch releases it.
        pool[0].report = Some(LoadReport {
            sampled_at_micros: 3_000,
            ..report(1, 0)
        });
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(ll.held(pool[0].id), 0);
        assert_eq!(scores, [1.0, 0.0]);
    }

    #[test]
    fn a_dark_fleet_scores_by_in_flight_and_a_missing_report_by_the_peer_baseline() {
        let ll = LeastLoad::new(0.15, 1024, 2000.0);
        let pool = [candidate(0, 3), candidate(1, 1)];
        let mut scores = [0.0f32; 2];
        ll.score(&RequestFacts::default(), &pool, &kept(2), &mut scores);
        assert_eq!(scores, [0.0, 1.0]);
        let mut mixed = pool;
        mixed[0].report = Some(LoadReport {
            live_rate: 2000.0,
            ..LoadReport::default()
        });
        mixed[0].in_flight = 0;
        ll.score(&RequestFacts::default(), &mixed, &kept(2), &mut scores);
        assert_eq!(scores[0], 1.0, "the idle reporting worker wins");
        assert!(
            scores[1] < 1.0,
            "the dark worker starts from the baseline plus its in-flight"
        );
    }

    #[test]
    fn power_of_two_takes_the_less_loaded_of_two_draws() {
        let pool = [candidate(0, 9), candidate(1, 9), candidate(2, 0)];
        let mut hits = 0;
        for seed in 0..100u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            PowerOfTwo.pick(&facts, &pool, &kept(3), &[0.0; 3], &mut out);
            assert_eq!(out.len(), 2);
            if out.first().map(|t| t.position) == Some(2) {
                hits += 1;
            }
        }
        // Worker 2 is drawn in two of three pairs and wins each time.
        assert!((50..=85).contains(&hits), "{hits}");
    }

    #[test]
    fn the_filters_read_their_bits() {
        let facts = RequestFacts::default();
        let cx = Constraints::default();
        let mut c = candidate(0, 0);
        assert!(Healthy.keep(&facts, &c, &cx));
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
        let tried = [c.id];
        let retry = RequestFacts {
            tried: &tried,
            ..facts
        };
        assert!(!NotTried.keep(&retry, &c, &cx));
        assert!(NotTried.soft() && HasRoom.soft() && !Healthy.soft());
    }
}
