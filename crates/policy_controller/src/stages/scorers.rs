//! The scorers written for no one policy. Every stateful one keeps its state
//! behind the lifecycle alone, and none allocates per decision.
//!
//! The credit of the sends since a worker's last report (`RankCredit`,
//! `Credits`) is in this file too. `ResidentWork` scores with it, and so does
//! the scorer of `least_load`, through the sent record in `tracking`.

use std::{
    cell::Cell,
    collections::VecDeque,
    sync::{
        atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxHashMap;

use crate::{
    document::spec::{CostMargin, Needs, Unreported},
    engine::{
        decision::{Decision, Overlap, Target},
        framework::{DecisionScope, ScoreFront, ScoreSpread, Scorer, TieBand},
        view::{Candidate, LoadReportCutoff, LoadRow, PoolFacts, RequestFacts, WorkerId},
        warm_up::WarmUpRule,
    },
    load::signals::{SignalSums, SignalTerm},
    stages::{
        cost_terms::{Coefficients, CostTerm},
        lifecycle::Lifecycle,
    },
    tracking::{
        dispatch_credit::{DispatchCounts, DispatchCredit},
        in_flight::{with_counts, Pieces, PoolInFlightCounts, WorkerSlot},
        lease::{EventKind, EventSet, Lease, Membership, Outcome, ReleaseReason, Side, Signal},
    },
};

/// Expected waits within this many seconds of the minimum are one tie: the
/// least-load scorer's own band, mapped onto the normalized scale per
/// decision. A wait is a 32-bit float, so past about 16 s it holds no
/// microsecond of its own: two waits that round together are a tie whatever
/// the band, and two that round apart are not.
pub(crate) const TIE_EPSILON_SECS: f32 = 1e-6;

/// A scorer that can read reports through a decision's cutoff. The bare
/// scorer reads the row directly, and the document wraps it only when it
/// sets a limit.
pub(crate) trait ReadsLoad: Scorer {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand;

    fn score_front_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        _kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> (TieBand, ScoreFront) {
        (
            self.score_with(facts, candidates, asked, out, load),
            ScoreFront::Unsaid,
        )
    }
}

#[derive(Debug)]
struct WithLoadReportAgeLimit<S> {
    stage: S,
    max_age_ms: u64,
}

pub(crate) fn with_load_report_age_limit(
    stage: impl ReadsLoad + 'static,
    max_age_ms: Option<u64>,
) -> Box<dyn Scorer> {
    match max_age_ms {
        Some(max_age_ms) => Box::new(WithLoadReportAgeLimit { stage, max_age_ms }),
        None => Box::new(stage),
    }
}

impl<S: ReadsLoad> Lifecycle for WithLoadReportAgeLimit<S> {
    fn on_decision_issued(&self, lease: &Lease, decision: &Decision) {
        self.stage.on_decision_issued(lease, decision);
    }

    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        self.stage
            .on_dispatched(lease, target, side, since_decision_micros);
    }

    fn on_completed(&self, lease: &Lease, outcome: &Outcome) {
        self.stage.on_completed(lease, outcome);
    }

    fn on_released(&self, lease: &Lease, reason: ReleaseReason) {
        self.stage.on_released(lease, reason);
    }

    fn on_membership(&self, change: &Membership<'_>) {
        self.stage.on_membership(change);
    }

    fn on_signal(&self, signal: &Signal<'_>) {
        self.stage.on_signal(signal);
    }

    fn keeps_loads(&self) -> bool {
        self.stage.keeps_loads()
    }

    fn needs_lifecycle(&self) -> bool {
        self.stage.needs_lifecycle()
    }

    fn hears(&self) -> EventSet {
        self.stage.hears()
    }
}

impl<S: ReadsLoad> Scorer for WithLoadReportAgeLimit<S> {
    fn name(&self) -> &'static str {
        self.stage.name()
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.stage
            .score_with(facts, candidates, kept, out, |row| cutoff.load(row))
    }

    fn score_front(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        asked: &[u32],
        out: &mut [f32],
    ) -> (TieBand, ScoreFront) {
        let cutoff = LoadReportCutoff::new(facts.now_micros, self.max_age_ms);
        self.stage
            .score_front_with(facts, candidates, kept, asked, out, |row| cutoff.load(row))
    }

    fn row_wise(&self) -> bool {
        self.stage.row_wise()
    }

    fn front_wise(&self) -> bool {
        self.stage.front_wise()
    }

    fn owns(&self) -> bool {
        self.stage.owns()
    }

    // The rows reach the stage as they stand, not through the cutoff: a
    // stage that owns its pool by a row's load would read past the limit.
    // No wrapped stage owns a pool today.
    fn own(
        &self,
        facts: &RequestFacts<'_>,
        pool: &PoolFacts<'_>,
        candidates: &[Candidate],
        keep: &mut dyn FnMut(&Candidate) -> bool,
        kept: &mut Vec<u32>,
    ) -> bool {
        self.stage.own(facts, pool, candidates, keep, kept)
    }

    fn spread(&self) -> ScoreSpread {
        self.stage.spread()
    }

    fn needs(&self) -> Needs {
        self.stage.needs()
    }

    fn scope(&self) -> Option<&dyn DecisionScope> {
        self.stage.scope()
    }

    fn refuses_by_reports(&self) -> bool {
        self.stage.refuses_by_reports()
    }

    fn warm_up_rule(&self) -> Option<WarmUpRule> {
        self.stage.warm_up_rule()
    }

    fn states_overlap(&self) -> bool {
        self.stage.states_overlap()
    }

    fn said(&self, facts: &RequestFacts<'_>, candidates: &[Candidate], row: u32) -> Overlap {
        self.stage.said(facts, candidates, row)
    }
}

/// The rows whose least and greatest cost are found by compares
/// ([`normalize_costs`]): the ranks of one engine, the rows of a draw.
const FEW_COSTS: usize = 16;

/// Lower raw costs become higher normalized scores; equal costs score `1.0`.
/// Returns the span of the costs, `0.0` when they are all equal.
pub(crate) fn normalize_costs(out: &mut [f32]) -> f32 {
    let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
    if out.len() <= FEW_COSTS {
        // A few rows, compared one by one: a cost that is no number moves
        // neither end, as the fold below leaves them.
        for &c in out.iter() {
            if c < lo {
                lo = c;
            }
            if c > hi {
                hi = c;
            }
        }
    } else {
        // Many rows, folded: the minimum and the maximum take any order,
        // so a step of the loop reads several rows.
        for &c in out.iter() {
            lo = lo.min(c);
            hi = hi.max(c);
        }
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
#[derive(Debug, Default)]
pub struct LeastRequests;

impl Lifecycle for LeastRequests {}

impl Scorer for LeastRequests {
    fn name(&self) -> &'static str {
        "least-requests"
    }

    // Scores by the count in flight: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        with_counts!(facts, |counts| {
            for (slot, &i) in out.iter_mut().zip(kept) {
                *slot = counts(&candidates[i as usize]) as f32;
            }
        });
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

    // Scores by the session's hit on a row.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        facts: &RequestFacts<'_>,
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
        // A pin that stands beside the rows names one row: it is looked for
        // once, and a decision under no such pin pays one test for it.
        let pinned = facts.pinned_at.checked_sub(1);
        if let Some(slot) = pinned.and_then(|at| kept.iter().position(|&i| i == at)) {
            out[slot] = 1.0;
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

/// Fewest session keys with a request in flight on the worker (the host's
/// count): the `min_group` assignment of a sticky session.
#[derive(Debug, Default)]
pub struct FewestSessions;

impl Lifecycle for FewestSessions {}

impl Scorer for FewestSessions {
    fn name(&self) -> &'static str {
        "fewest-sessions"
    }

    // Scores by the sessions a row holds: a late count moves the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = f32::from(candidates[i as usize].active_sessions);
        }
        normalize_costs(out);
        TieBand::NORMALIZED
    }

    fn needs(&self) -> Needs {
        Needs {
            active_sessions: true,
            ..Needs::default()
        }
    }
}

/// The dispatches credited to one rank since its last load report, oldest
/// first within each lane. What a decision reads (the held token-work and
/// requests) rides
/// one atomic word, so deciding threads read a row without a lock. The
/// dispatches themselves sit in [`DISPATCH_LANES`] lanes, each behind its own
/// lock and fed by the threads mapped to it: concurrent dispatches to one row
/// (deciders that read the same minimum book the same rank) rarely share a
/// lock. `oldest_micros` tells a decision whether its report's sample reaches
/// any held dispatch without taking one.
#[derive(Debug)]
pub(crate) struct RankCredit {
    /// The held token-work in the high bits and the held requests in the low
    /// [`REQUEST_BITS`]: one word, so a reader never sees a torn pair.
    held: AtomicU64,
    /// How many of the held dispatches were recorded at no tokens: those
    /// of a lease that states no size, in a count of sends. As wide as the
    /// held word's count of requests, which bounds it. A word of its own,
    /// written behind the held word at a record and lowered ahead of it
    /// at a release and at a retirement, so it never counts a dispatch
    /// the held word does not. The held word's subtraction there is a
    /// `Release` and the read of both words loads the held word with
    /// `Acquire`, so a reader finds that order on every target. A read
    /// loads the two words apart: across a record, a release or a
    /// retirement it may see one without the other for the dispatch in
    /// flight.
    unsized_held: AtomicU16,
    /// The estimates of the held dispatches that stated no size, summed:
    /// what a reader that prices such a send by its text adds to the
    /// stated token-work ([`Self::counts_through`]). A word of its own,
    /// kept as the count beside it is: raised behind the held word at a
    /// record, and lowered ahead of it at a release and at a retirement,
    /// whose subtraction from the held word is a `Release`. A read loads
    /// the held word with `Acquire` and this one behind it. Across a
    /// record it may see a dispatch in one word and not in the other: its
    /// request without its estimate, or the estimate of a dispatch that
    /// landed behind the read of the held word. Across a release or a
    /// retirement it may miss the estimate of a dispatch the held word
    /// still counts, and reads that send at no tokens: it never adds the
    /// estimate of a dispatch the held word has let go.
    estimated_held: AtomicU64,
    /// The lowest of the lanes' front instants; `u64::MAX` when none. A
    /// dispatch lowers it under its lane's lock; a release recomputes it
    /// under every lane's lock, so it never reads above any lane's front.
    oldest_micros: AtomicU64,
    lanes: [DispatchLane; DISPATCH_LANES],
    /// What a stage that keeps the row's cost as one number was last told
    /// of the row. A report's tell, its first decision and a dispatch hold
    /// it while they write that number, and such a stage records its
    /// dispatches under it, so each number is of one report and one
    /// credit.
    told: Mutex<Told>,
    /// The sample instant of the report whose number such a stage keeps
    /// of the row, `u64::MAX` where it keeps none. Written with the told
    /// report held. A decision that costs the row from its parts reads it
    /// with no lock: under a later sample its release may take dispatches
    /// that number counts ([`ResidentWork::settle`]).
    kept_of: AtomicU64,
}

/// The last report a stage was told of one rank row, with the row's slot:
/// what the number the stage keeps of the row is made of, beside its credit.
/// The number is kept from the report's first decision on
/// ([`ResidentWork::settle`]).
/// An entry that names no slot writes no number: one told no report yet,
/// one the tally let go, and one whose cell another entry was told.
#[derive(Debug, PartialEq, Eq)]
struct Told {
    slot: WorkerSlot,
    sampled_at_micros: u64,
    row: Option<LoadRow>,
}

impl Told {
    /// Whether `word` is the report this one holds, under the same slot.
    fn says(&self, word: &Self) -> bool {
        self.slot == word.slot
            && self.sampled_at_micros == word.sampled_at_micros
            && self.row == word.row
    }
}

impl Default for Told {
    fn default() -> Self {
        Self {
            slot: WorkerSlot::NONE,
            sampled_at_micros: 0,
            row: None,
        }
    }
}

/// What a lane holds of one dispatch's size: the tokens its lease stated,
/// or, for a lease that stated none, an estimate of them. The mark of an
/// estimate is the top bit of the size's own 32, so a size is held under
/// that bit at the write, stated or estimated: a size of 2^31 and more is
/// held as the largest under it. A stated one is then neither read as an
/// estimate nor taken out of the estimates' word.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct SendSize(u32);

impl SendSize {
    /// The mark of an estimate.
    const ESTIMATE: u32 = 1 << 31;
    /// The largest size held, stated or estimated.
    const MOST: u32 = Self::ESTIMATE - 1;

    /// A dispatch whose lease stated `tokens`. One of no tokens stated no
    /// size: it is held as an estimate of none.
    pub(crate) const fn stated(tokens: u32) -> Self {
        match tokens {
            0 => Self(Self::ESTIMATE),
            tokens if tokens > Self::MOST => Self(Self::MOST),
            tokens => Self(tokens),
        }
    }

    /// A dispatch whose lease stated no token count, at `estimate` tokens.
    pub(crate) const fn estimated(estimate: u32) -> Self {
        let held = if estimate > Self::MOST {
            Self::MOST
        } else {
            estimate
        };
        Self(Self::ESTIMATE | held)
    }

    /// The size of `lease`'s dispatch in a count of sends: the tokens it
    /// states, and for a lease that states none its text's bytes over four
    /// as an estimate ([`Lease::prompt_size`]). A lease a frame's one count
    /// made states that count.
    pub(crate) const fn of(lease: &Lease) -> Self {
        match lease.tokens {
            0 => Self::estimated(lease.prompt_size()),
            tokens => Self::stated(tokens),
        }
    }

    /// Whether the dispatch stated no size.
    const fn is_unsized(self) -> bool {
        self.0 & Self::ESTIMATE != 0
    }

    /// The tokens the dispatch stated: none for one that stated no size.
    const fn tokens(self) -> u32 {
        if self.is_unsized() {
            0
        } else {
            self.0
        }
    }

    /// The estimate of a dispatch that stated no size: none for one that
    /// stated its size.
    const fn estimate(self) -> u32 {
        if self.is_unsized() {
            self.0 & Self::MOST
        } else {
            0
        }
    }
}

/// A lane's dispatches: the instant and the size of each, oldest first.
/// The one type that holds them: a record holds a dispatch here, and a
/// release and a retirement let the oldest go. Its lane's lock guards it.
#[derive(Debug, Default)]
struct Dispatches(VecDeque<(u64, SendSize)>);

impl Dispatches {
    /// Holds one dispatch of `size` made at `at_micros`, and says whether
    /// it is the oldest held now.
    ///
    /// It lands at its instant's place: a dispatch that waited for its
    /// lane's lock, or one a stepped clock dated earlier, lands behind a
    /// later one, and the release and the retirement read a lane from its
    /// oldest. Instants mostly rise, so the newest is asked
    /// first: a dispatch no older than it is a push, and the place is
    /// searched for the one that is.
    fn hold(&mut self, at_micros: u64, size: SendSize) -> bool {
        let newest = self.0.back().map(|&(held_at, _)| held_at);
        if newest.is_some_and(|newest| newest > at_micros) {
            let at = self.0.partition_point(|&(held_at, _)| held_at <= at_micros);
            self.0.insert(at, (at_micros, size));
            at == 0
        } else {
            self.0.push_back((at_micros, size));
            newest.is_none()
        }
    }

    /// The instant of the oldest dispatch held.
    fn oldest(&self) -> Option<u64> {
        self.0.front().map(|&(at, _)| at)
    }

    /// Lets the oldest dispatch go where it was made at or before
    /// `through_micros`, and hands back its size.
    fn let_go(&mut self, through_micros: u64) -> Option<SendSize> {
        let &(at, size) = self.0.front()?;
        if at > through_micros {
            return None;
        }
        self.0.pop_front();
        Some(size)
    }

    /// How many dispatches it holds.
    fn len(&self) -> usize {
        self.0.len()
    }
}

/// One lane of a row's dispatches, on a cache line of its own: the threads
/// mapped to it append here, and its traffic never invalidates another lane.
#[derive(Debug)]
#[repr(align(64))]
struct DispatchLane {
    dispatches: Mutex<Dispatches>,
    /// The instant at this lane's front; `u64::MAX` when none.
    oldest_micros: AtomicU64,
}

/// Lanes per rank row. A deciding thread keeps the lane it drew first; a
/// host with more deciding threads than lanes shares them.
const DISPATCH_LANES: usize = 8;

/// Dispatches kept per rank row between reports, over its lanes: a dispatch
/// that lands past the bound lets the row's oldest go, by instant across
/// the lanes, so the newest credits stay and the row holds the bound: rows
/// that stand there read level, and a decision spreads its sends over
/// them. The row holds at most the bound and one dispatch per deciding
/// thread, so the request count's [`REQUEST_BITS`] bits never fill.
pub(crate) const RANK_CREDIT_KEPT: u64 = 8_192;

/// The low bits of [`RankCredit::held`] that count requests: room for the
/// kept dispatches, with the token-work above them.
const REQUEST_BITS: u32 = 16;
const REQUEST_MASK: u64 = (1 << REQUEST_BITS) - 1;

/// The two numbers a held word says ([`RankCredit::held`]), each as wide
/// as the word: what the crate's own passes price.
#[derive(Clone, Copy)]
pub(crate) struct HeldCredit {
    /// The token-work of the held dispatches that stated a size.
    pub(crate) tokens: u64,
    /// The held dispatches.
    pub(crate) requests: u64,
}

impl HeldCredit {
    /// Of a row the count holds no entry of.
    const NONE: Self = Self {
        tokens: 0,
        requests: 0,
    };
}

/// The three numbers a row's read hands a stage that prices a send of no
/// stated size itself ([`RankCredit::sent_through`]): three loads a row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SentCredit {
    /// The held dispatches.
    pub(crate) requests: u32,
    /// The token-work of the held dispatches that stated a size.
    pub(crate) tokens: u64,
    /// How many of the held dispatches stated no size.
    pub(crate) unsized_requests: u32,
}

thread_local! {
    /// The lane this thread's dispatches take, drawn once from a shared
    /// position: deciding threads spread over the lanes in arrival order.
    static LANE: Cell<Option<usize>> = const { Cell::new(None) };
}

static NEXT_LANE: AtomicUsize = AtomicUsize::new(0);

fn lane_of_this_thread() -> usize {
    LANE.with(|lane| {
        lane.get().unwrap_or_else(|| {
            let drawn = NEXT_LANE.fetch_add(1, Ordering::Relaxed) % DISPATCH_LANES;
            lane.set(Some(drawn));
            drawn
        })
    })
}

/// Pins this thread's lane, so a test owns one lane per thread.
#[cfg(test)]
pub(crate) fn take_lane(lane: usize) {
    LANE.with(|drawn| drawn.set(Some(lane)));
}

impl Default for DispatchLane {
    fn default() -> Self {
        Self {
            dispatches: Mutex::default(),
            oldest_micros: AtomicU64::new(u64::MAX),
        }
    }
}

impl DispatchLane {
    /// Under the largest value where the lane holds a dispatch: that value
    /// says an empty lane alone, to a retirement that reads the fronts with
    /// no lock.
    fn publish_oldest(&self, dispatches: &Dispatches) {
        let oldest = dispatches
            .oldest()
            .map_or(u64::MAX, |at| at.min(u64::MAX - 1));
        self.oldest_micros.store(oldest, Ordering::Release);
    }
}

impl Default for RankCredit {
    fn default() -> Self {
        Self {
            held: AtomicU64::new(0),
            unsized_held: AtomicU16::new(0),
            estimated_held: AtomicU64::new(0),
            oldest_micros: AtomicU64::new(u64::MAX),
            lanes: Default::default(),
            told: Mutex::new(Told::default()),
            kept_of: AtomicU64::new(u64::MAX),
        }
    }
}

impl RankCredit {
    /// One dispatch as the held word counts it.
    fn weight(tokens: u32) -> u64 {
        u64::from(tokens) << REQUEST_BITS | 1
    }

    /// Holds one dispatch of `size` made at `at_micros`. A dispatch that
    /// stated no size is counted as one, and its estimate is kept beside
    /// the stated token-work.
    pub(crate) fn record(&self, at_micros: u64, size: SendSize) {
        let lane = &self.lanes[lane_of_this_thread()];
        let mut dispatches = lane.dispatches.lock();
        // The lane's front moves only where the dispatch lands there.
        let front = dispatches.hold(at_micros, size);
        let held = self
            .held
            .fetch_add(Self::weight(size.tokens()), Ordering::Relaxed);
        if size.is_unsized() {
            self.unsized_held.fetch_add(1, Ordering::Relaxed);
            self.estimated_held
                .fetch_add(u64::from(size.estimate()), Ordering::Relaxed);
        }
        if front {
            lane.publish_oldest(&dispatches);
        }
        // Under the lane's lock: a release, which holds every lane, never
        // recomputes the row's oldest past this dispatch. Dispatch instants
        // mostly rise, so this is a read.
        if at_micros < self.oldest_micros.load(Ordering::Acquire) {
            self.oldest_micros.fetch_min(at_micros, Ordering::AcqRel);
        }
        drop(dispatches);
        // The dispatch found the row at its bound: the row's oldest goes
        // for it, by instant across the lanes, so the newest credits stay,
        // whichever lane holds them, past a partial release.
        if held & REQUEST_MASK >= RANK_CREDIT_KEPT {
            self.retire_oldest();
        }
    }

    /// Takes a dispatch of `tokens` its lane let go out of the held word.
    /// The subtraction is a `Release`: a read that loads the word with
    /// `Acquire` finds every write made ahead of it, the count of the
    /// dispatches of no tokens among them ([`Self::sent_through`]).
    fn take_out(&self, tokens: u32) {
        self.held.fetch_sub(Self::weight(tokens), Ordering::Release);
    }

    /// Every lane's dispatches under their locks, taken in lane order: the
    /// one order of every hold of them. A retirement holds one lane's lock
    /// alone, so it waits on none of them for a second lane.
    fn lock_lanes(&self) -> [MutexGuard<'_, Dispatches>; DISPATCH_LANES] {
        std::array::from_fn(|i| self.lanes[i].dispatches.lock())
    }

    /// Lets the row's oldest dispatch go for one that landed past the
    /// bound: the front of the lane that publishes the lowest instant,
    /// under that lane's lock alone. Each dispatch that lands past the
    /// bound lets one go, so the row holds the bound again once they are
    /// through. A row a release took to the bound in the meantime keeps
    /// its dispatches: the bound is read and the dispatch taken out in one
    /// step of the held word, so retirements under two lanes' locks never
    /// let two go for one. The row's oldest instant stays where it was: a
    /// value too low costs a release walk, never a missed one.
    fn retire_oldest(&self) {
        loop {
            // Each lane's front as the lane published it, read with no
            // lock; the lowest lane of the lowest instant.
            let (oldest, lane) = self
                .lanes
                .iter()
                .fold((u64::MAX, &self.lanes[0]), |low, lane| {
                    let front = lane.oldest_micros.load(Ordering::Acquire);
                    if front < low.0 {
                        (front, lane)
                    } else {
                        low
                    }
                });
            let mut dispatches = lane.dispatches.lock();
            // The bound read again under the lock: a release since this
            // dispatch landed may have taken the row to it, and then the
            // newer credits stay.
            if self.held.load(Ordering::Relaxed) & REQUEST_MASK <= RANK_CREDIT_KEPT {
                return;
            }
            // Under the lock the lane's published front is its own. One
            // past the instant read means the one read went to another
            // dispatch's retirement, and the lanes are read again.
            let front = lane.oldest_micros.load(Ordering::Relaxed);
            if front > oldest {
                continue;
            }
            let Some(&(_, size)) = dispatches.0.front() else {
                continue;
            };
            // The estimate and the count of no stated size first and the
            // held word next, as a release lowers them. The bound and the
            // subtraction are one step of that word: a retirement under
            // another lane's lock reads the same word, and one of the two
            // takes the last dispatch past the bound.
            if size.is_unsized() {
                self.estimated_held
                    .fetch_sub(u64::from(size.estimate()), Ordering::Relaxed);
                self.unsized_held.fetch_sub(1, Ordering::Relaxed);
            }
            // A case reads the row here, between the two writes.
            #[cfg(test)]
            if let Some(read) = tests::BETWEEN_A_RETIREMENTS_WRITES.get() {
                read(self);
            }
            let weight = Self::weight(size.tokens());
            let taken = self
                .held
                .try_update(Ordering::Release, Ordering::Relaxed, |held| {
                    (held & REQUEST_MASK > RANK_CREDIT_KEPT).then(|| held - weight)
                });
            if taken.is_err() {
                if size.is_unsized() {
                    self.unsized_held.fetch_add(1, Ordering::Relaxed);
                    self.estimated_held
                        .fetch_add(u64::from(size.estimate()), Ordering::Relaxed);
                }
                return;
            }
            dispatches.0.pop_front();
            lane.publish_oldest(&dispatches);
            return;
        }
    }

    /// Whether a stage keeps the number of a report sampled before
    /// `sampled_at_micros`: a decision that costs the row from its parts
    /// under that later sample may release dispatches the number counts.
    #[inline]
    fn keeps_before(&self, sampled_at_micros: u64) -> bool {
        sampled_at_micros > self.kept_of.load(Ordering::Acquire)
    }

    /// The held requests and their tokens, stated or estimated, once the
    /// dispatches made at or before `through_micros` are released: a
    /// report sampled then reflects them, or a time limit ends them. What
    /// a document's `sent.` terms read: a send that stated no token count
    /// adds its estimate. Inlined into a reader's row loop: three loads
    /// and a compare a row, the row's oldest instant, the held word and
    /// the estimates' word, with the release behind a call. The held word
    /// is loaded with `Acquire`, against the `Release` of a release's and
    /// a retirement's subtraction: the estimates are read as they stood at
    /// that write or later, so no read adds the estimate of a dispatch
    /// the held word has let go, on any target.
    #[inline(always)]
    pub(crate) fn counts_through(&self, through_micros: u64) -> DispatchCounts {
        if self.reaches(through_micros) {
            self.release_through(through_micros);
        }
        let held = self.held.load(Ordering::Acquire);
        DispatchCounts {
            // The word's low bits: the narrower type cuts none of them.
            requests: (held & REQUEST_MASK) as u32,
            tokens: (held >> REQUEST_BITS) + self.estimated(),
        }
    }

    /// The held requests and stated token-work of [`Self::counts_through`]
    /// with the count of the held dispatches of no tokens in the
    /// estimates' place: three loads too. The held word is loaded with
    /// `Acquire`, against the `Release` of a release's and a retirement's
    /// subtraction: the count is read as it stood at that write or later,
    /// so no read finds more dispatches of no tokens than dispatches
    /// across a release, on any target.
    #[inline(always)]
    pub(crate) fn sent_through(&self, through_micros: u64) -> SentCredit {
        if self.reaches(through_micros) {
            self.release_through(through_micros);
        }
        let held = self.held.load(Ordering::Acquire);
        SentCredit {
            requests: (held & REQUEST_MASK) as u32,
            tokens: held >> REQUEST_BITS,
            unsized_requests: u32::from(self.unsized_held.load(Ordering::Relaxed)),
        }
    }

    /// The estimates of the held dispatches that stated no size, as the
    /// word stands: read behind the held word ([`Self::estimated_held`]).
    #[inline(always)]
    pub(crate) fn estimated(&self) -> u64 {
        self.estimated_held.load(Ordering::Relaxed)
    }

    /// Calls `each` with the instant and the stated tokens of every
    /// dispatch the row holds once those made at or before
    /// `through_micros` are released: the dispatches
    /// [`Self::counts_through`] counts. Lane by
    /// lane, oldest first within a lane. The release and the calls run
    /// under one hold of every lane's lock, so the row stands still from
    /// the release on: no dispatch made at or before the instant lands
    /// between the two, and one recorded meanwhile waits for its lane and
    /// lands behind the read.
    ///
    /// The release runs where the row's oldest instant reaches
    /// `through_micros`, read under the hold. A dispatch lowers that
    /// instant under its lane's lock, a release stores it under every
    /// lane's and a retirement leaves it. So past `through_micros` it says
    /// that no lane holds a dispatch the release would take, and such a
    /// read stores nothing on the line every decision loads for the row.
    pub(crate) fn each_through(&self, through_micros: u64, mut each: impl FnMut(u64, u32)) {
        let mut held_lanes = self.lock_lanes();
        if self.reaches(through_micros) {
            self.release_held(&mut held_lanes, through_micros);
        }
        for &(at_micros, size) in held_lanes.iter().flat_map(|dispatches| &dispatches.0) {
            each(at_micros, size.tokens());
        }
    }

    /// Whether the row holds a dispatch made at or before the sample: one
    /// a report sampled then reflects, which its release takes.
    #[inline(always)]
    fn reaches(&self, sampled_at_micros: u64) -> bool {
        self.oldest_micros.load(Ordering::Acquire) <= sampled_at_micros
    }

    /// The held token-work and requests, as the word stands.
    #[inline(always)]
    fn held_credit(&self) -> HeldCredit {
        Self::credit_of(self.held.load(Ordering::Relaxed))
    }

    /// The token-work and requests a held word says.
    #[inline(always)]
    const fn credit_of(held: u64) -> HeldCredit {
        HeldCredit {
            tokens: held >> REQUEST_BITS,
            requests: held & REQUEST_MASK,
        }
    }

    /// Releases every lane's dispatches made at or before the sample and
    /// recomputes the row's oldest, under every lane's lock in lane order,
    /// so no dispatch lands between the lanes' reads and the store. Hands
    /// back the held word as it stands once the locks are free: a read of
    /// two numbers takes them from it and keeps nothing of the row across
    /// the call. Out of line: a decision reads a row far more often than
    /// a report releases from it, and with this body inlined the read was
    /// too large to stay in a second scorer's row loop.
    #[cold]
    #[inline(never)]
    fn release_through(&self, sampled_at_micros: u64) -> u64 {
        let mut held_lanes = self.lock_lanes();
        self.release_held(&mut held_lanes, sampled_at_micros);
        // Loaded behind the locks, as a reader's own load is: under them
        // the value would be carried across the unlocking.
        drop(held_lanes);
        self.held.load(Ordering::Relaxed)
    }

    /// [`Self::release_through`] under the hold of every lane's lock its
    /// caller keeps: the release itself and the row's oldest instant.
    /// Inlined by force into the release, whose body it was.
    #[inline(always)]
    fn release_held(
        &self,
        held_lanes: &mut [MutexGuard<'_, Dispatches>; DISPATCH_LANES],
        sampled_at_micros: u64,
    ) {
        // No dispatch lands under every lane's lock, so the count of the
        // held dispatches of no tokens stands still here: a row that
        // holds none releases by the loop it ran before that count was
        // kept.
        if self.unsized_held.load(Ordering::Relaxed) != 0 {
            self.release_marked(held_lanes, sampled_at_micros);
        } else {
            for (dispatches, lane) in held_lanes.iter_mut().zip(&self.lanes) {
                while let Some(size) = dispatches.let_go(sampled_at_micros) {
                    self.take_out(size.tokens());
                }
                lane.publish_oldest(dispatches);
            }
        }
        let oldest = self
            .lanes
            .iter()
            .map(|lane| lane.oldest_micros.load(Ordering::Acquire))
            .min()
            .unwrap_or(u64::MAX);
        self.oldest_micros.store(oldest, Ordering::Release);
    }

    /// The release of a row that holds a dispatch of no tokens: each one
    /// that goes leaves the estimates' word and that count first and the
    /// held word next, so a read that crosses the release misses the
    /// price of one dispatch at most.
    #[cold]
    #[inline(never)]
    fn release_marked(
        &self,
        held_lanes: &mut [MutexGuard<'_, Dispatches>; DISPATCH_LANES],
        sampled_at_micros: u64,
    ) {
        for (dispatches, lane) in held_lanes.iter_mut().zip(&self.lanes) {
            while let Some(size) = dispatches.let_go(sampled_at_micros) {
                if size.is_unsized() {
                    self.estimated_held
                        .fetch_sub(u64::from(size.estimate()), Ordering::Relaxed);
                    self.unsized_held.fetch_sub(1, Ordering::Relaxed);
                }
                self.take_out(size.tokens());
            }
            lane.publish_oldest(dispatches);
        }
    }

    /// Dispatches held over every lane, for tests and dumps.
    fn len(&self) -> usize {
        self.lanes
            .iter()
            .map(|lane| lane.dispatches.lock().len())
            .sum()
    }
}

// What a test in another file reads of a row. The row's words stay this
// file's own.
#[cfg(test)]
impl RankCredit {
    /// The held token-work and requests, and the sends of no stated size.
    pub(crate) fn held_for_test(&self) -> (HeldCredit, u16) {
        (
            self.held_credit(),
            self.unsized_held.load(Ordering::Relaxed),
        )
    }

    /// The row's three words as they stand: the stated token-work, the
    /// requests, the sends of no stated size and their estimates.
    pub(crate) fn words_for_test(&self) -> (u64, u64, u16, u64) {
        let (HeldCredit { tokens, requests }, unsized_held) = self.held_for_test();
        (tokens, requests, unsized_held, self.estimated())
    }

    /// One retirement, as a dispatch that found the row at its bound runs
    /// it.
    pub(crate) fn retire_for_test(&self) {
        self.retire_oldest();
    }

    /// Whether every lane's lock is held now: what a dispatch's record
    /// and a release wait for.
    pub(crate) fn lanes_held_for_test(&self) -> bool {
        self.lanes.iter().all(|lane| lane.dispatches.is_locked())
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

/// The since-report dispatch credit of a load scorer: per rank row, the
/// dispatches no report reflects yet. A decision and a dispatch to a rank the
/// tally knows load the map without a shared write (a read-write lock's
/// reader count is one, and eight deciders queued on it); a rank's first
/// dispatch and a membership change replace the map whole, which is rare.
/// Within a row the dispatches take per-thread lanes: deciders that read the
/// same minimum book the same rank, and one lock there queued them too.
#[derive(Debug)]
pub(crate) struct Credits {
    pub(crate) tally: ArcSwap<FxHashMap<RankKey, Arc<RankCredit>>>,
}

impl Default for Credits {
    fn default() -> Self {
        Self {
            tally: ArcSwap::from_pointee(FxHashMap::default()),
        }
    }
}

impl Credits {
    /// Dispatches no report reflects yet on `worker`, over every rank of it.
    pub(crate) fn held(&self, worker: WorkerId) -> usize {
        self.tally
            .load()
            .iter()
            .filter(|((id, _), _)| *id == worker)
            .map(|(_, since)| since.len())
            .sum()
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub(crate) fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.tally
            .load()
            .get(&(worker, dp_rank))
            .map_or(0, |since| since.len())
    }

    /// The entry of `rank`, entered where the tally did not know the rank.
    /// `None` where a membership change took the rank away meanwhile.
    fn entry(&self, rank: RankKey) -> Option<Arc<RankCredit>> {
        if let Some(since) = self.tally.load().get(&rank) {
            return Some(Arc::clone(since));
        }
        self.tally.rcu(|tally| {
            if tally.contains_key(&rank) {
                return Arc::clone(tally);
            }
            let mut next = FxHashMap::clone(tally);
            next.insert(rank, Arc::new(RankCredit::default()));
            Arc::new(next)
        });
        self.tally.load().get(&rank).map(Arc::clone)
    }

    /// The tally without the ranks `gone` names: the map replaced whole, as a
    /// membership change is rare and a decision never waits on it. Each
    /// entry that went is shown to `dropped` once the tally finds it no
    /// more: a word that reached it earlier may still hold it.
    fn forget(&self, gone: impl Fn(&RankKey) -> bool, mut dropped: impl FnMut(&RankCredit)) {
        let mut went: Vec<Arc<RankCredit>> = Vec::new();
        self.tally.rcu(|tally| {
            went.clear();
            let mut next = FxHashMap::clone(tally);
            next.retain(|rank, since| {
                let stays = !gone(rank);
                if !stays {
                    went.push(Arc::clone(since));
                }
                stays
            });
            next
        });
        for since in &went {
            dropped(since);
        }
    }

    /// Credits the rank a dispatch landed on. `credit` is handed the
    /// rank's entry with the dispatch's instant and its size
    /// ([`SendSize::of`]): the tokens its lease states, and where it
    /// states none its text's bytes over four as an estimate. `credit`
    /// records it ([`RankCredit::record`]). A count of sends records the
    /// size as it is, a profile's and the one a `least-load` built by
    /// hand keeps: the entry counts a send of no stated tokens as one of
    /// no stated size and keeps its estimate, and each reader prices it.
    /// `resident-work` keeps a tally of its own and records its mean
    /// there: a credit of no tokens would read its worker as free until
    /// that worker's next report. It records the dispatch under the
    /// report it was told, as it keeps the rank's number.
    ///
    /// A rank the tally does not know is entered only where `present`
    /// says its row still stands, so a late dispatch enters no rank
    /// behind a removal. A rank the tally knows takes the dispatch
    /// unasked: a removal forgets the worker's entries.
    pub(crate) fn dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl FnOnce() -> bool,
        credit: impl FnOnce(&RankCredit, u64, SendSize),
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
        // A rank the tally knows is credited on its own entry; the first
        // dispatch to a rank enters it (another dispatch may have entered it
        // meanwhile: the entry the map holds is the one credited).
        if let Some(since) = self.tally.load().get(&rank) {
            credit(since, dispatched_micros, SendSize::of(lease));
            return;
        }
        if !present() {
            return;
        }
        if let Some(since) = self.entry(rank) {
            credit(&since, dispatched_micros, SendSize::of(lease));
        }
    }

    /// Drops the ranks a membership change takes away, each entry that
    /// goes shown to `dropped` first.
    pub(crate) fn membership(&self, change: &Membership<'_>, dropped: impl FnMut(&RankCredit)) {
        match change {
            // Every rank of the worker goes with it. A resized worker's ranks
            // are a new set: what a vanished rank held would never release,
            // and a kept rank's next report is one interval away. Rare, so a
            // walk over the tally is cheaper than a second index per worker.
            Membership::Removed(id)
            | Membership::Replaced { old: id, .. }
            | Membership::Resized(id) => {
                self.forget(|(worker, _)| worker == id, dropped);
            }
            // A snapshot (every worker of every pool, after a resync) stands
            // for removals that may never have arrived: only its ids keep a
            // tally. Rare, so a sorted copy is cheaper than a scan per row.
            Membership::Snapshot(ids) => {
                let mut sorted: Vec<u64> = ids.iter().map(|w| w.0).collect();
                sorted.sort_unstable();
                self.forget(
                    |(worker, _)| sorted.binary_search(&worker.0).is_err(),
                    dropped,
                );
            }
            Membership::Added(_) | Membership::Relabelled(_) | Membership::Left { .. } => {}
        }
    }
}

/// The mean prefill a load scorer prices a request of unknown size at when
/// its profile names none.
pub const DEFAULT_MEAN_PREFILL_TOKENS: u32 = 1024;

/// The weight of the KV-pressure barrier in a load scorer's expected wait
/// when its profile names none.
pub const DEFAULT_KV_PRESSURE_WEIGHT: f32 = 0.15;

/// The drain rate, in tokens per second, a load scorer assumes when no
/// candidate reports a live one and its profile names none.
pub const DEFAULT_THROUGHPUT: f32 = 2000.0;

/// A cost slot not written yet: a candidate without a report is costed
/// after the reporting peers' baseline is known. Costs are never negative.
pub(crate) const PENDING: f32 = f32::NEG_INFINITY;

/// The resident work, in context tokens, lower better: what a worker's
/// report says its running requests hold, the prompts dispatched to it since
/// that report (the credit the least-load scorer books too), and
/// `request_tokens` for every request it holds: the reported ones, the
/// credited ones, and the ones the host holds beyond both, those priced at
/// the worker's mean context as well. Every row handed is scored: every
/// survivor of a whole pass, or the rows a picker drew. A worker
/// whose report does not count resident tokens (the row's
/// `resident_counted`), or that has no report, scores as the best reporting
/// peer among the rows handed plus its in-flight requests, each at the mean
/// prefill and `request_tokens`; with no row handed counting them, the
/// in-flight count decides.
#[derive(Debug)]
pub struct ResidentWork {
    pub request_tokens: u32,
    pub mean_prefill_tokens: u32,
    credits: Credits,
    /// The cost of each member the stage was told a report of, by the
    /// member's slot ([`Kept`]).
    kept: Pieces<Kept>,
    /// The entry that writes each cell, by the cell's slot: the one last
    /// told a report under the slot. A tell and a membership change hold
    /// this while they give a cell to an entry or take it from one, under
    /// that entry's told report as well. One entry at a time names a slot,
    /// so a cell's writes are one entry's, one at a time. A dispatch takes
    /// no hold of this: it writes the cell the entry it credits names,
    /// under that entry's told report.
    bound: Mutex<Bound>,
}

/// The entry bound to each slot's cell.
type Bound = FxHashMap<WorkerSlot, Arc<RankCredit>>;

/// A member's resident work as one number: the cost of the report its
/// stage was told and of its credit, with no request beyond the two, and
/// how many requests that accounts for. The stage writes it at the first
/// decision under a report it was told and where it credits a dispatch
/// behind that decision, one write at a time,
/// and a decision reads it by the member's slot with no lock. A write
/// turns the cell twice, before it and behind it: a read that finds the
/// turn odd, or moved between its first load and its last, is handed
/// nothing, so no read pairs one report's stamp with another's number.
/// Each store releases and each load acquires: a read that met any store
/// of a write meets that write's first turn at its last load.
///
/// A row is handed the number by the stamp of its report alone, so a
/// cell keeps one report an instant: none of an instant it was told two
/// reports under, and none of an instant an entry that wrote it before
/// was told a report of ([`Kept::keep_none_through`]). A row then reads
/// a number only where it holds the very report the number is of, where
/// its host told the stage each world before a decision read it.
#[derive(Debug, Default)]
struct Kept {
    /// One more at each write's start and at its end: odd while a write
    /// is at the cell.
    turn: AtomicU64,
    /// The sample instant of the report the number is of. Zero where the
    /// cell keeps none.
    stamp: AtomicU64,
    /// The cost's bits in the low half and the requests it accounts for
    /// in the high half.
    word: AtomicU64,
    /// The cell keeps no number of a report sampled at this instant or
    /// before it. Its writers' alone: no decision reads it.
    after: AtomicU64,
    /// The sample instant of a report the stage was told and keeps no
    /// number of yet: the number is kept from the first decision that
    /// costs a row by the report ([`ResidentWork::settle`]). Zero where
    /// the cell awaits no decision. A decision reads it only where the
    /// cell hands its row no number.
    awaited: AtomicU64,
}

impl Kept {
    /// The number kept of the report `row` holds and the requests it
    /// accounts for: `None` where the cell keeps another report's or none,
    /// and where a write crossed the read.
    #[inline]
    fn read(&self, row: &Candidate) -> Option<(f32, u32)> {
        let turn = self.turn.load(Ordering::Acquire);
        let stamp = self.stamp.load(Ordering::Acquire);
        let word = self.word.load(Ordering::Acquire);
        if turn & 1 == 1 || self.turn.load(Ordering::Acquire) != turn {
            return None;
        }
        if stamp == 0 || stamp != row.load_sampled_at_micros || row.load.is_none() {
            return None;
        }
        Some((f32::from_bits(word as u32), (word >> 32) as u32))
    }

    /// Whether the cell awaits the first decision under the report `row`
    /// holds.
    #[inline]
    fn awaits(&self, row: &Candidate) -> bool {
        let awaited = self.awaited.load(Ordering::Acquire);
        awaited != 0 && awaited == row.load_sampled_at_micros
    }

    /// Keeps no number, and awaits the first decision under a report
    /// sampled at `sampled_at_micros`: no decision where that is zero.
    fn await_decision(&self, sampled_at_micros: u64) {
        self.clear();
        self.awaited.store(sampled_at_micros, Ordering::Release);
    }

    /// Whether the cell keeps a number of a report sampled at
    /// `sampled_at_micros`: of none stamped zero, as no report is.
    fn takes(&self, sampled_at_micros: u64) -> bool {
        sampled_at_micros > self.after.load(Ordering::Relaxed)
    }

    /// Keeps `cost` as the number of the report sampled at
    /// `sampled_at_micros`, accounting for `accounted` requests: no number
    /// under the stamp zero. One write at a time: the caller holds the
    /// told report of the entry bound to the cell
    /// ([`ResidentWork::bound`]).
    fn write(&self, sampled_at_micros: u64, cost: f32, accounted: u64) {
        let accounted = u32::try_from(accounted).unwrap_or(u32::MAX);
        let word = u64::from(accounted) << 32 | u64::from(cost.to_bits());
        let turn = self.turn.load(Ordering::Relaxed);
        self.turn.store(turn.wrapping_add(1), Ordering::Release);
        self.stamp.store(sampled_at_micros, Ordering::Release);
        self.word.store(word, Ordering::Release);
        self.turn.store(turn.wrapping_add(2), Ordering::Release);
    }

    /// Keeps no number: no write where the cell keeps none already.
    fn clear(&self) {
        if self.stamp.load(Ordering::Relaxed) != 0 {
            self.write(0, 0.0, 0);
        }
    }

    /// Keeps no number, and none after of a report sampled at
    /// `sampled_at_micros` or before it. Called where the entry that
    /// wrote the cell lets it go, with the newest instant that entry was
    /// told, and where a second report is told under one instant: a row
    /// stamped that instant or an earlier one may hold another report
    /// than the one the cell is told next.
    fn keep_none_through(&self, sampled_at_micros: u64) {
        let after = self.after.load(Ordering::Relaxed);
        self.after
            .store(after.max(sampled_at_micros), Ordering::Relaxed);
        self.await_decision(0);
    }
}

impl ResidentWork {
    pub fn new(request_tokens: u32) -> Self {
        Self {
            request_tokens,
            mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            credits: Credits::default(),
            kept: Pieces::default(),
            bound: Mutex::new(Bound::default()),
        }
    }

    /// Dispatches no report reflects yet on one rank of `worker`.
    pub fn held_at(&self, worker: WorkerId, dp_rank: u16) -> usize {
        self.credits.held_at(worker, dp_rank)
    }

    /// The resident work of a candidate whose report counts resident tokens,
    /// with its since-report credit.
    fn reported_cost(
        &self,
        in_flight: u32,
        r: &LoadRow,
        credit_tokens: u64,
        credit_requests: u64,
    ) -> f32 {
        let reported = u64::from(r.running) + u64::from(r.queued);
        let unaccounted = u64::from(in_flight).saturating_sub(reported + credit_requests);
        let resident = r.resident_tokens as f32;
        let mean_context = if reported > 0 {
            resident / reported as f32
        } else {
            self.mean_prefill_tokens as f32
        };
        let requests = reported + credit_requests + unaccounted;
        resident
            + credit_tokens as f32
            + unaccounted as f32 * mean_context
            + self.request_tokens as f32 * requests as f32
    }
}

impl ResidentWork {
    /// Costs the rows the first pass left pending, those with no report
    /// that counts resident tokens: the best reporting peer among the rows
    /// handed plus the row's requests in flight, or the requests alone
    /// where no row reports. Out of line: a pool whose every row reports
    /// pays for no second pass.
    #[cold]
    #[inline(never)]
    fn cost_pending(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        fleet_reports: bool,
    ) {
        let peer_baseline = if fleet_reports {
            out.iter()
                .copied()
                .filter(|cost| *cost != PENDING)
                .fold(f32::INFINITY, f32::min)
        } else {
            0.0
        };
        let in_flight_work = self.mean_prefill_tokens as f32 + self.request_tokens as f32;
        with_counts!(facts, |counts| {
            for (slot, &i) in out.iter_mut().zip(kept) {
                if *slot == PENDING {
                    let in_flight = counts(&candidates[i as usize]) as f32;
                    *slot = if fleet_reports {
                        peer_baseline + in_flight * in_flight_work
                    } else {
                        in_flight
                    };
                }
            }
        });
    }
}

impl ResidentWork {
    /// Writes into `cell` the number of the report `told` holds and of
    /// `held`, the rank's credit as its held word reads. Called with the
    /// entry's told report held, where the row holds no dispatch that
    /// report reflects: the word is then the credit a decision reads
    /// behind its release, and the number is of one report and one
    /// credit. A report that counts no resident tokens, no report, and a
    /// report of an instant the cell keeps none of, keep none.
    fn keep(&self, cell: &Kept, told: &Told, held: u64) {
        match told.row.as_ref() {
            Some(row) if row.resident_counted && cell.takes(told.sampled_at_micros) => {
                let (tokens, requests) = (held >> REQUEST_BITS, held & REQUEST_MASK);
                let reported = u64::from(row.running) + u64::from(row.queued);
                let cost = self.reported_cost(0, row, tokens, requests);
                cell.write(told.sampled_at_micros, cost, reported + requests);
            }
            _ => cell.clear(),
        }
    }

    /// What a decision that costed `row` from its parts owes the number
    /// of the row's rank: one of two things, each rare.
    ///
    /// The first decision that costs a row by the report the entry was
    /// told keeps the report's number in `cell` from here on. It has just
    /// released the dispatches the report reflects, as on a stage told
    /// nothing. So every release is a decision's, here as there, and the
    /// row holds what it holds there whatever row is costed from its parts
    /// next: a late report's, or one of a world that stood before. The
    /// number is kept only where the row holds no dispatch the report
    /// reflects: one that landed behind this decision's release waits for
    /// the next decision's, and the cell awaits that one. It is kept under
    /// every lane's lock, so another decision's release is whole before
    /// it or whole behind it.
    ///
    /// A decision under a later sample than the kept report's sends the
    /// number back: its release may have taken dispatches the number
    /// counts, so the cell awaits the kept report's next decision, which
    /// reads the row as it stands. A world a decision still holds can show
    /// such a sample: a member that left and came back is told an earlier
    /// one.
    ///
    /// Cold and out of line: a report comes here once, or once more
    /// behind a dispatch it reflects or a later sample.
    #[cold]
    #[inline(never)]
    fn settle(&self, cell: Option<&Kept>, since: &RankCredit, row: &Candidate, slot: WorkerSlot) {
        let told = since.told.lock();
        let kept_of = since.kept_of.load(Ordering::Relaxed);
        if kept_of != u64::MAX {
            if row.load_sampled_at_micros > kept_of {
                since.kept_of.store(u64::MAX, Ordering::Release);
                if let Some(kept) = self.kept.cell(told.slot) {
                    kept.await_decision(told.sampled_at_micros);
                }
            }
            return;
        }
        let its = told.says(&Told {
            slot,
            sampled_at_micros: row.load_sampled_at_micros,
            row: row.load,
        });
        let Some(cell) = cell.filter(|_| its) else {
            return;
        };
        let lanes = since.lock_lanes();
        if since.oldest_micros.load(Ordering::Acquire) > told.sampled_at_micros {
            since
                .kept_of
                .store(told.sampled_at_micros, Ordering::Release);
            self.keep(cell, &told, since.held.load(Ordering::Relaxed));
            cell.awaited.store(0, Ordering::Release);
        }
        drop(lanes);
    }

    /// [`ResidentWork::settle`] for the first decision under a report:
    /// the rank's entry is found again here, once a report a rank, so the
    /// routine that costs a row hands back its cost alone.
    #[cold]
    #[inline(never)]
    fn settle_first(
        &self,
        tally: &FxHashMap<RankKey, Arc<RankCredit>>,
        cell: Option<&Kept>,
        row: &Candidate,
        slot: WorkerSlot,
    ) {
        if let Some(since) = tally.get(&(row.id, row.dp_rank)) {
            self.settle(cell, since, row, slot);
        }
    }

    /// Takes `word`, a rank's report as its host tells it, into `since`,
    /// the rank's entry, and leaves `cell`, the cell of the word's slot,
    /// awaiting the report's first decision. Called with the bound entries
    /// held.
    ///
    /// An entry told a slot it does not name is bound to the slot's cell.
    /// The entry that wrote the cell before lets it go: the member's rank
    /// as it stood before a registration moved it, which decisions made
    /// before the move still credit. A report older than the one the entry
    /// holds is late and dropped. The report it holds, told again, is no
    /// word: a host tells every rank at each of its publications. Another
    /// report under the instant of the one it holds is taken, and the cell
    /// keeps no number of that instant.
    ///
    /// A tell releases nothing, reads nothing of the row's dispatches and
    /// keeps no number. A host tells a world's reports before it shows the
    /// world, and a decision may still read the world that stands, or meet
    /// a late report, at any time after. A stage told nothing releases what
    /// a report reflects at the first decision that costs a row by it, and
    /// each of those rows is costed right only while the told stage
    /// releases there too. So the cell awaits that decision
    /// ([`ResidentWork::settle`]), and until it each row is costed from
    /// its parts, as on a stage told nothing.
    fn tell(&self, bound: &mut Bound, cell: &Kept, since: &Arc<RankCredit>, word: Told) {
        let mut told = since.told.lock();
        let mut news = false;
        if told.slot != word.slot {
            // A host that moved the rank's slot: the entry lets the cell
            // it wrote go.
            if let Some(left) = self.kept.cell(told.slot) {
                left.keep_none_through(told.sampled_at_micros);
                bound.remove(&told.slot);
            }
            let former = bound.insert(word.slot, Arc::clone(since));
            if let Some(former) = former.filter(|former| !Arc::ptr_eq(former, since)) {
                let mut theirs = former.told.lock();
                cell.keep_none_through(theirs.sampled_at_micros);
                theirs.slot = WorkerSlot::NONE;
                former.kept_of.store(u64::MAX, Ordering::Release);
            }
            told.slot = word.slot;
            news = true;
        }
        if word.sampled_at_micros >= told.sampled_at_micros && !told.says(&word) {
            if word.sampled_at_micros == told.sampled_at_micros {
                cell.keep_none_through(word.sampled_at_micros);
            }
            *told = word;
            news = true;
        }
        if news {
            // No number yet: the cell awaits the report's first decision.
            since.kept_of.store(u64::MAX, Ordering::Release);
            cell.await_decision(told.sampled_at_micros);
        }
    }

    /// The cost of `row` by its report and its credit under one load of
    /// the tally, or the mark of a row costed after the reporting peers'
    /// baseline is known. A report releases the dispatches it reflects
    /// whatever the score reads of them: a rank whose report counts no
    /// resident tokens would hold its credit at the bound. The row's
    /// requests in flight are asked for where the row is costed, and of no
    /// row that is marked. Inlined into each row loop of the score: as a
    /// call it read 19 instructions a row more for a host that lends no
    /// counts.
    ///
    /// A release rides with its test, whichever view the row came in:
    /// under a later sample than the report whose number the stage keeps
    /// it may have taken dispatches that number counts, and the number is
    /// sent back ([`ResidentWork::release`]). One stage can be handed rows
    /// beside a lent table and rows a host wrote.
    #[inline(always)]
    fn cost_reported(
        &self,
        tally: &FxHashMap<RankKey, Arc<RankCredit>>,
        row: &Candidate,
        report: Option<&LoadRow>,
        in_flight: impl FnOnce() -> u32,
        reporting: &mut usize,
    ) -> f32 {
        let Some(r) = report else {
            return PENDING;
        };
        let credit = tally.get(&(row.id, row.dp_rank));
        let HeldCredit { tokens, requests } = credit.map_or(HeldCredit::NONE, |since| {
            // Only a release that takes a dispatch can leave a kept
            // number stale, so the test rides that release: a row that
            // holds none at or before its sample reads the word alone.
            if since.reaches(row.load_sampled_at_micros) {
                return self.release(since, row);
            }
            since.held_credit()
        });
        if r.resident_counted {
            *reporting += 1;
            self.reported_cost(in_flight(), r, tokens, requests)
        } else {
            PENDING
        }
    }

    /// Releases the dispatches `row`'s report reflects, as a stage told
    /// nothing does, and sends back a number kept of an earlier report:
    /// the release may have taken dispatches that number counts
    /// ([`ResidentWork::settle`]). Hands back the held token-work and
    /// requests behind the release, which settling does not write, so
    /// the pass keeps nothing of the row across this call. Cold and out
    /// of line: a decision reads a row far more often than a report
    /// releases from it.
    #[cold]
    #[inline(never)]
    fn release(&self, since: &RankCredit, row: &Candidate) -> HeldCredit {
        let held = since.release_through(row.load_sampled_at_micros);
        if since.keeps_before(row.load_sampled_at_micros) {
            self.settle(None, since, row, WorkerSlot::NONE);
        }
        RankCredit::credit_of(held)
    }
}

impl ResidentWork {
    /// The rows' costs beside a lent table, and how many of them report.
    /// A row whose member's number the stage keeps is costed by one read
    /// of it, where the row holds no request beyond those the number
    /// accounts for. Any other row is costed by its report and its
    /// credit, under one load of the tally taken at the first such row,
    /// or marked and costed after the reporting peers' baseline is known.
    /// A row with no report is marked at once: no count, no cell and no
    /// tally is read for it. A cell holds a number only of a report that
    /// counts resident tokens, so a row with any other report is asked for
    /// none. Inlined into the score: as a call it read 19 to 37
    /// instructions a decision more.
    #[inline(always)]
    fn cost_lent(
        &self,
        lent: &PoolInFlightCounts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> usize {
        let mut reporting = 0usize;
        let run = lent.run();
        let first = self.kept.first();
        let mut tally = None;
        for (slot, &i) in out.iter_mut().zip(kept) {
            let c = &candidates[i as usize];
            let report = load(c);
            *slot = match report {
                None => PENDING,
                Some(counted) if counted.resident_counted => {
                    let in_flight = run.of(c);
                    let cell = self.kept.cell_from(first, run.slot_of(c));
                    match cell.and_then(|cell| cell.read(c)) {
                        Some((cost, accounted)) if in_flight <= accounted => {
                            reporting += 1;
                            cost
                        }
                        _ => {
                            let tally = tally.get_or_insert_with(|| self.credits.tally.load());
                            let cost =
                                self.cost_reported(tally, c, report, || in_flight, &mut reporting);
                            // The report's first decision keeps the
                            // rank's number from here on.
                            if cell.is_some_and(|cell| cell.awaits(c)) {
                                self.settle_first(tally, cell, c, run.slot_of(c));
                            }
                            cost
                        }
                    }
                }
                Some(_) => {
                    let tally = tally.get_or_insert_with(|| self.credits.tally.load());
                    self.cost_reported(tally, c, report, || run.of(c), &mut reporting)
                }
            };
        }
        reporting
    }

    /// The rows' costs by each row's own count, for a host that lends
    /// none, and how many of them report: every row by its report and its
    /// credit under one load of the tally, as before a number was kept.
    /// Out of line: inside the score, beside the other loop, it read two
    /// instructions a row more than it did before.
    #[inline(never)]
    fn cost_own(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> usize {
        let mut reporting = 0usize;
        let tally = self.credits.tally.load();
        for (slot, &i) in out.iter_mut().zip(kept) {
            let c = &candidates[i as usize];
            *slot = self.cost_reported(&tally, c, load(c), || c.in_flight, &mut reporting);
        }
        reporting
    }
}

impl Lifecycle for ResidentWork {
    fn on_dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
    ) {
        // The credit and the number of the rank it landed on, under the
        // report the rank's entry was told: a number is written of one
        // report and one credit.
        self.credits.dispatched(
            lease,
            target,
            side,
            since_decision_micros,
            || true,
            |since, at_micros, size| {
                // The stage's own tally holds a send of no stated size at
                // the stage's mean.
                let tokens = match size.tokens() {
                    0 => self.mean_prefill_tokens,
                    stated => stated,
                };
                let told = since.told.lock();
                since.record(at_micros, SendSize::stated(tokens));
                // A number is kept from its report's first decision on:
                // ahead of it a dispatch writes none.
                if since.kept_of.load(Ordering::Relaxed) == u64::MAX {
                    return;
                }
                let Some(cell) = self.kept.cell(told.slot) else {
                    return;
                };
                if at_micros > told.sampled_at_micros {
                    self.keep(cell, &told, since.held.load(Ordering::Relaxed));
                } else {
                    // The told report reflects this dispatch. A stage
                    // told nothing releases it at the next decision under
                    // the report: the cell awaits that decision again.
                    since.kept_of.store(u64::MAX, Ordering::Release);
                    cell.await_decision(told.sampled_at_micros);
                }
            },
        );
    }

    fn on_membership(&self, change: &Membership<'_>) {
        // A rank that goes takes its number with it, and its entry names
        // no cell after: a dispatch that still holds it writes none.
        let mut bound = self.bound.lock();
        self.credits.membership(change, |since| {
            let mut told = since.told.lock();
            if let Some(cell) = self.kept.cell(told.slot) {
                cell.keep_none_through(told.sampled_at_micros);
                bound.remove(&told.slot);
            }
            *told = Told::default();
            since.kept_of.store(u64::MAX, Ordering::Release);
        });
    }

    /// A rank's report as its host shows it to decisions: the stage keeps
    /// the rank's cost as one number in the cell of its slot. A host that
    /// names no slot is kept no number.
    fn on_signal(&self, signal: &Signal<'_>) {
        let Signal::Load {
            worker,
            dp_rank,
            row,
            slot,
            sampled_at_micros,
            ..
        } = *signal
        else {
            return;
        };
        let Some(cell) = self.kept.lay(slot) else {
            return;
        };
        // No membership change takes the rank's entry away meanwhile.
        let mut bound = self.bound.lock();
        if let Some(since) = self.credits.entry((worker, dp_rank)) {
            let word = Told {
                slot,
                sampled_at_micros,
                row: Some(row),
            };
            self.tell(&mut bound, cell, &since, word);
        }
    }

    fn keeps_loads(&self) -> bool {
        true
    }

    /// The dispatch: a credit is counted from it.
    fn hears(&self) -> EventSet {
        EventSet::NONE.with(EventKind::Dispatched)
    }
}

impl Scorer for ResidentWork {
    fn name(&self) -> &'static str {
        "resident-work"
    }

    // Costs by the count in flight and its credits: late ones move the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    /// The rows a picker drew are costed among themselves, as the
    /// least-load scorer costs them: a row with no report, or whose report
    /// counts no resident tokens, is costed against the rows handed.
    fn row_wise(&self) -> bool {
        true
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
            loads: true,
            ..Needs::default()
        }
    }
}

impl ReadsLoad for ResidentWork {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        // A loop a kind of host. The loop beside a lent table stands here;
        // the loop of a host that lends no counts is a call, and stays the
        // one it was before a number was kept. Each reads a row's report
        // through `load`: a row past a decision's cutoff has none, so no
        // cell and no tally is read for it.
        let reporting = match facts.lent_in_flight {
            Some(lent) => self.cost_lent(lent, candidates, kept, out, load),
            None => self.cost_own(candidates, kept, out, load),
        };
        if reporting < kept.len() {
            self.cost_pending(facts, candidates, kept, out, reporting > 0);
        }
        // The least-load scorer's band, on a score in tokens: equal work is
        // a tie, drawn by the picker.
        let span = normalize_costs(out);
        if span > 0.0 {
            TieBand((TIE_EPSILON_SECS / span).min(1.0))
        } else {
            TieBand::NORMALIZED
        }
    }
}

/// The sum of a row's named fields times their coefficients. Rows within
/// the cost margin of the least cost share the top score.
#[derive(Debug)]
pub struct LeastCost {
    terms: Coefficients,
    margin: CostMargin,
    request_cost: Option<f64>,
    loads: bool,
}

impl LeastCost {
    pub(crate) const NAME: &str = "least-cost";

    pub(crate) fn new(
        terms: Box<[(CostTerm, f64)]>,
        margin: CostMargin,
        unreported: Option<Unreported>,
    ) -> Self {
        let loads = terms.iter().any(|&(term, _)| term.reads_load());
        let request_cost = match unreported {
            Some(Unreported::OverLeast { request_cost }) => Some(request_cost),
            Some(Unreported::MissingAsZero) | None => None,
        };
        Self {
            terms: Coefficients::of(&terms),
            margin,
            request_cost,
            loads,
        }
    }

    /// Prices the rows left open over the least kept sum, then scores the
    /// costs between their ends.
    #[inline(always)]
    fn price_and_score(
        &self,
        out: &mut [f32],
        costs: &mut [RowCost],
        mut bounds: CostBounds,
        pending: bool,
    ) -> TieBand {
        if pending {
            let baseline = if bounds.least.is_finite() {
                bounds.least
            } else {
                0.0
            };
            for cost in costs.iter_mut() {
                if let RowCost::OverLeast(price) = *cost {
                    let sum = baseline + price;
                    bounds.include(sum);
                    *cost = RowCost::Sum(sum);
                }
            }
        }
        write_cost_scores(out, self.margin, costs, bounds)
    }

    /// The cost of a row with its report and count as the decision reads them.
    #[inline(always)]
    fn cost_of<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        load: Option<&LoadRow>,
        reads_count: bool,
        count: impl Fn() -> u32,
    ) -> RowCost {
        let in_flight = reads_count.then(&count);
        let sum = 0.0 + f64::from(in_flight.unwrap_or_default()) * self.terms.in_flight;
        let missing = || {
            self.request_cost.map_or(RowCost::Sum(sum), |price| {
                RowCost::OverLeast(f64::from(in_flight.unwrap_or_else(count)) * price)
            })
        };
        match load {
            Some(report) => self
                .terms
                .add_report::<RESIDENT, WAITING, KV_OR_RUNNING>(
                    sum,
                    report,
                    self.request_cost.is_some(),
                )
                .map_or_else(missing, RowCost::Sum),
            None if self.loads => missing(),
            None => RowCost::Sum(sum),
        }
    }

    #[inline(always)]
    fn sum_rows<const RESIDENT: bool, const WAITING: bool, const KV_OR_RUNNING: bool>(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
        counts: impl Fn(&Candidate) -> u32,
        reads_count: bool,
    ) -> (CostBounds, bool) {
        let mut bounds = CostBounds::EMPTY;
        let mut pending = false;
        costs.extend(kept.iter().map(|&i| {
            let row = &candidates[i as usize];
            let cost =
                self.cost_of::<RESIDENT, WAITING, KV_OR_RUNNING>(load(row), reads_count, || {
                    counts(row)
                });
            match cost {
                RowCost::Sum(sum) => bounds.include(sum),
                RowCost::OverLeast(_) => pending = true,
            }
            cost
        }));
        (bounds, pending)
    }

    /// A document names which optional values the walk tests.
    #[inline(always)]
    fn walk<const KV_OR_RUNNING: bool>(
        &self,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
        counts: impl Fn(&Candidate) -> u32,
        reads_count: bool,
    ) -> (CostBounds, bool) {
        let resident = self.terms.resident_tokens != 0.0;
        let waiting = self.terms.waiting_uncached_tokens != 0.0;
        match (resident, waiting) {
            (false, false) => self.sum_rows::<false, false, KV_OR_RUNNING>(
                candidates,
                kept,
                costs,
                load,
                counts,
                reads_count,
            ),
            (false, true) => self.sum_rows::<false, true, KV_OR_RUNNING>(
                candidates,
                kept,
                costs,
                load,
                counts,
                reads_count,
            ),
            (true, false) => self.sum_rows::<true, false, KV_OR_RUNNING>(
                candidates,
                kept,
                costs,
                load,
                counts,
                reads_count,
            ),
            (true, true) => self.sum_rows::<true, true, KV_OR_RUNNING>(
                candidates,
                kept,
                costs,
                load,
                counts,
                reads_count,
            ),
        }
    }

    fn collect_costs(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        costs: &mut Vec<RowCost>,
        load: impl Fn(&Candidate) -> Option<&LoadRow>,
    ) -> (CostBounds, bool) {
        // A written count is a field of the row. Beside a lent table, read
        // it where a term or the missing-report price needs it.
        with_counts!(facts, |counts, written| {
            let reads_count = written || self.terms.in_flight != 0.0;
            if self.terms.names_kv_or_running() {
                self.walk::<true>(candidates, kept, costs, load, counts, reads_count)
            } else {
                self.walk::<false>(candidates, kept, costs, load, counts, reads_count)
            }
        })
    }
}

impl Lifecycle for LeastCost {}

impl Scorer for LeastCost {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    // Costs by reports and the count in flight: late ones move the pick.
    fn refuses_by_reports(&self) -> bool {
        false
    }

    fn row_wise(&self) -> bool {
        true
    }

    fn needs(&self) -> Needs {
        Needs {
            loads: self.loads,
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCost {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (bounds, pending) = self.collect_costs(facts, candidates, kept, costs, load);
            self.price_and_score(out, costs, bounds, pending)
        })
    }
}

/// `least-cost` where the document names a `sent.` term: the sum of
/// [`LeastCost`], then the profile's sends. A document that names none is
/// built as [`LeastCost`] alone and reads no count.
#[derive(Debug)]
pub(crate) struct LeastCostWithDispatches {
    cost: LeastCost,
    sent: Arc<DispatchCredit>,
}

impl LeastCostWithDispatches {
    pub(crate) fn new(cost: LeastCost, sent: Arc<DispatchCredit>) -> Self {
        Self { cost, sent }
    }
}

impl Lifecycle for LeastCostWithDispatches {}

impl Scorer for LeastCostWithDispatches {
    fn name(&self) -> &'static str {
        self.cost.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.cost.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.cost.row_wise()
    }

    // A report ends the sends before it, so the sends need the reports
    // though no term may read one. A term that prices the sends by their
    // tokens needs each request's size stated: a send of text alone is
    // counted by its text's bytes.
    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            prompt_size: self.cost.terms.sizes_sends(),
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCostWithDispatches {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (_, pending) = self
                .cost
                .collect_costs(facts, candidates, kept, costs, load);
            // The sends close the sum: their names sort behind every field
            // the walk adds. A row priced over the least kept no sum and
            // takes none. The sends move the ends of the sums, so the ends
            // are taken here and the walk's are dropped.
            let terms = &self.cost.terms;
            let sends = self.sent.pass(facts);
            let mut bounds = CostBounds::EMPTY;
            for (&row, cost) in kept.iter().zip(costs.iter_mut()) {
                if let RowCost::Sum(sum) = cost {
                    let sent = sends.counts(&candidates[row as usize]);
                    terms.add_sent(sum, sent);
                    bounds.include(*sum);
                }
            }
            self.cost.price_and_score(out, costs, bounds, pending)
        })
    }
}

/// `least-cost` where the document names a `signal.` term: the sum of
/// [`LeastCost`], then the profile's sends where a term names them, then
/// the signals in name order. A row with no fresh value for a signal a
/// term names is unreported for the scorer: under `over-least` it is
/// priced over the least by its requests in flight, as a row with no
/// report is, and under `missing-as-zero` the term adds nothing. A
/// document that names no signal is built as one of the other two and
/// reads no cell.
#[derive(Debug)]
pub(crate) struct LeastCostSignals {
    cost: LeastCost,
    /// The profile's count of sends, where a term reads it.
    sent: Option<Arc<DispatchCredit>>,
    signals: Box<[SignalTerm]>,
}

impl LeastCostSignals {
    pub(crate) fn new(
        cost: LeastCost,
        sent: Option<Arc<DispatchCredit>>,
        signals: Box<[SignalTerm]>,
    ) -> Self {
        Self {
            cost,
            sent,
            signals,
        }
    }
}

impl Lifecycle for LeastCostSignals {}

impl Scorer for LeastCostSignals {
    fn name(&self) -> &'static str {
        self.cost.name()
    }

    fn refuses_by_reports(&self) -> bool {
        self.cost.refuses_by_reports()
    }

    fn row_wise(&self) -> bool {
        self.cost.row_wise()
    }

    // A report ends the sends before it, so the sends need the reports
    // though no term may read one. A term that prices the sends by their
    // tokens needs each request's size stated. The cells stand beside the
    // rows of a profile whose stage says it reads a signal.
    fn needs(&self) -> Needs {
        Needs {
            loads: self.cost.loads || self.sent.is_some(),
            prompt_size: self.cost.terms.sizes_sends(),
            signal_cells: true,
            ..Needs::default()
        }
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
}

impl ReadsLoad for LeastCostSignals {
    fn score_with(
        &self,
        facts: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
        load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
    ) -> TieBand {
        with_costs(|costs| {
            let (_, mut pending) = self
                .cost
                .collect_costs(facts, candidates, kept, costs, load);
            // The sends close the fields' sum and the signals close the
            // sends': `signal.` sorts behind every other name. A row priced
            // over the least kept no sum and takes none. Both move the ends
            // of the sums, so the ends are taken here and the walk's are
            // dropped.
            let terms = &self.cost.terms;
            let sends = self.sent.as_ref().map(|sent| sent.pass(facts));
            // Once a pass: the facts' cells and each signal's oldest
            // instant still read. Once a row: the row's cells.
            let signals = SignalSums::new(facts, &self.signals);
            let mut bounds = CostBounds::EMPTY;
            with_counts!(facts, |counts| {
                for (&i, cost) in kept.iter().zip(costs.iter_mut()) {
                    let RowCost::Sum(sum) = cost else {
                        continue;
                    };
                    let row = &candidates[i as usize];
                    if let Some(sends) = &sends {
                        let sent = sends.counts(row);
                        terms.add_sent(sum, sent);
                    }
                    let (priced, whole) = signals.add(*sum, signals.row(row));
                    *sum = priced;
                    // A signal with no fresh value leaves the row open under
                    // `over-least`, as no report does in the walk: its sum
                    // is dropped and its requests in flight price it.
                    match self.cost.request_cost.filter(|_| !whole) {
                        Some(price) => {
                            *cost = RowCost::OverLeast(f64::from(counts(row)) * price);
                            pending = true;
                        }
                        None => bounds.include(priced),
                    }
                }
            });
            self.cost.price_and_score(out, costs, bounds, pending)
        })
    }
}

#[derive(Clone, Copy)]
enum RowCost {
    Sum(f64),
    /// The price for the row's requests in flight, before adding the least reported cost.
    OverLeast(f64),
}

impl RowCost {
    fn value(self) -> f64 {
        match self {
            Self::Sum(value) | Self::OverLeast(value) => value,
        }
    }
}

/// The ends of the finite costs found while pricing the rows.
struct CostBounds {
    least: f64,
    greatest: f64,
}

impl CostBounds {
    const EMPTY: Self = Self {
        least: f64::INFINITY,
        greatest: f64::NEG_INFINITY,
    };

    fn include(&mut self, cost: f64) {
        if cost < self.least {
            self.least = cost;
        }
        if cost > self.greatest {
            self.greatest = cost;
        }
    }
}

thread_local! {
    /// One slot per row scored. The deciding thread keeps the buffer for reuse.
    static COSTS: Cell<Vec<RowCost>> = const { Cell::new(Vec::new()) };
}

fn with_costs(work: impl FnOnce(&mut Vec<RowCost>) -> TieBand) -> TieBand {
    // Move the buffer out so a cost callback can score another set of rows.
    let mut costs = COSTS.take();
    let band = work(&mut costs);
    costs.clear();
    COSTS.set(costs);
    band
}

/// Writes scores from finite costs with a finite span, lower cost better.
/// Costs within `margin` of the least cost score `1.0`. The returned band
/// is zero, since the tie is already in the scores. Calls `cost` once for
/// each position of `out`, in order.
pub fn score_costs(
    out: &mut [f32],
    margin: CostMargin,
    mut cost: impl FnMut(usize) -> f64,
) -> TieBand {
    with_costs(|costs| {
        costs.extend((0..out.len()).map(|i| RowCost::Sum(cost(i))));
        let mut bounds = CostBounds::EMPTY;
        for &cost in costs.iter() {
            bounds.include(cost.value());
        }
        write_cost_scores(out, margin, costs, bounds)
    })
}

fn write_cost_scores(
    out: &mut [f32],
    margin: CostMargin,
    costs: &[RowCost],
    CostBounds { least, greatest }: CostBounds,
) -> TieBand {
    let span = greatest - least;
    if span == 0.0 {
        out.fill(1.0);
    } else {
        let limit = least + margin.absolute + margin.relative * least.abs();
        for (slot, &cost) in out.iter_mut().zip(costs) {
            let cost = cost.value();
            *slot = if cost <= limit {
                1.0
            } else {
                // Keep rounding to f32 from adding a row to the tie.
                ((1.0 - (cost - least) / span) as f32).min(1.0_f32.next_down())
            };
        }
    }
    TieBand(0.0)
}

#[cfg(test)]
pub(crate) mod tests {
    thread_local! {
        /// What a case runs on its own thread between a retirement's two
        /// writes, behind the estimates' and the count's and ahead of the
        /// held word's: the row as a read finds it there.
        pub(crate) static BETWEEN_A_RETIREMENTS_WRITES: Cell<Option<fn(&RankCredit)>> =
            const { Cell::new(None) };
    }

    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::{
        document::spec::{LeastCostParams, WarmupSliceParams},
        engine::{
            decision::{Pick, Targets},
            framework::{Picker, Shortlist},
            view::{Health, LoadRow},
        },
        load::ReportSignals,
        policies::{
            cache_aware::{Gates, WarmupSlice},
            least_load::LeastLoad,
            power_of_two::PowerOfTwoCountFirst,
        },
        stages::pickers::{target_of, Best, CountFirst},
        tracking::{
            in_flight::tests::{Lent, RUN_READERS},
            lease::{EventKind, TransportClass},
        },
        Draw,
    };

    pub(crate) fn candidate(position: u32, in_flight: u32) -> Candidate {
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

    pub(crate) fn kept(n: u32) -> Vec<u32> {
        (0..n).collect()
    }

    #[test]
    fn a_report_age_limit_preserves_the_stages_methods() {
        #[derive(Debug, Default)]
        struct ProbeState {
            events: Mutex<Vec<EventKind>>,
        }

        impl DecisionScope for ProbeState {
            fn open(&self, _: &RequestFacts<'_>, _: &[Candidate], _: &PoolFacts<'_>) {}
            fn close(&self) {}
        }

        #[derive(Clone, Debug, Default)]
        struct Probe(Arc<ProbeState>);

        impl Lifecycle for Probe {
            fn on_decision_issued(&self, _: &Lease, _: &Decision) {
                self.0.events.lock().push(EventKind::Issued);
            }
            fn on_dispatched(&self, _: &Lease, _: &Target, _: Side, _: u32) {
                self.0.events.lock().push(EventKind::Dispatched);
            }
            fn on_completed(&self, _: &Lease, _: &Outcome) {
                self.0.events.lock().push(EventKind::Completed);
            }
            fn on_released(&self, _: &Lease, _: ReleaseReason) {
                self.0.events.lock().push(EventKind::Released);
            }
            fn on_membership(&self, _: &Membership<'_>) {
                self.0.events.lock().push(EventKind::Membership);
            }
            fn on_signal(&self, _: &Signal<'_>) {
                self.0.events.lock().push(EventKind::Signal);
            }
            fn keeps_loads(&self) -> bool {
                true
            }
            fn needs_lifecycle(&self) -> bool {
                true
            }
        }

        impl Scorer for Probe {
            fn name(&self) -> &'static str {
                "probe"
            }
            fn score(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                out: &mut [f32],
            ) -> TieBand {
                self.score_with(facts, candidates, kept, out, |row| row.load.as_ref())
            }
            fn score_front(
                &self,
                facts: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                asked: &[u32],
                out: &mut [f32],
            ) -> (TieBand, ScoreFront) {
                self.score_front_with(facts, candidates, kept, asked, out, |row| row.load.as_ref())
            }
            fn row_wise(&self) -> bool {
                true
            }
            fn front_wise(&self) -> bool {
                true
            }
            fn owns(&self) -> bool {
                true
            }
            fn own(
                &self,
                _: &RequestFacts<'_>,
                _: &PoolFacts<'_>,
                candidates: &[Candidate],
                keep: &mut dyn FnMut(&Candidate) -> bool,
                kept: &mut Vec<u32>,
            ) -> bool {
                kept.extend(
                    candidates
                        .iter()
                        .filter(|row| keep(row))
                        .map(|row| row.position),
                );
                !kept.is_empty()
            }
            fn spread(&self) -> ScoreSpread {
                ScoreSpread::Levels(1.0)
            }
            fn needs(&self) -> Needs {
                Needs {
                    loads: true,
                    ..Needs::default()
                }
            }
            fn scope(&self) -> Option<&dyn DecisionScope> {
                Some(self.0.as_ref())
            }
            fn refuses_by_reports(&self) -> bool {
                false
            }
            fn warm_up_rule(&self) -> Option<WarmUpRule> {
                WarmupSlice::new(&WarmupSliceParams::default(), Gates::default()).warm_up_rule()
            }
            fn states_overlap(&self) -> bool {
                true
            }
            fn said(&self, _: &RequestFacts<'_>, _: &[Candidate], row: u32) -> Overlap {
                Overlap {
                    blocks: 3,
                    of: 5 + row as u16,
                    ..Overlap::default()
                }
            }
        }

        impl ReadsLoad for Probe {
            fn score_with(
                &self,
                _: &RequestFacts<'_>,
                candidates: &[Candidate],
                kept: &[u32],
                out: &mut [f32],
                load: impl Fn(&Candidate) -> Option<&LoadRow> + Copy,
            ) -> TieBand {
                for (slot, &index) in out.iter_mut().zip(kept) {
                    *slot = if load(&candidates[index as usize]).is_some() {
                        1.0
                    } else {
                        0.0
                    };
                }
                TieBand(0.0625)
            }
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
                (band, ScoreFront::of(self.spread(), out))
            }
        }

        let bare = Probe::default();
        let wrapped = with_load_report_age_limit(bare.clone(), Some(37));
        let facts = RequestFacts {
            now_micros: 71_337,
            ..RequestFacts::default()
        };
        let mut rows = [candidate(0, 0), candidate(1, 0)];
        rows[1].load = Some(LoadRow::default());
        rows[1].load_sampled_at_micros = facts.now_micros;
        let score = |stage: &dyn Scorer| {
            let mut out = [0.0; 2];
            let band = stage.score(&facts, &rows, &[0, 1], &mut out);
            (band, out)
        };
        let score_front = |stage: &dyn Scorer| {
            let mut out = [0.0];
            let front = stage.score_front(&facts, &rows, &[0, 1], &[1], &mut out);
            (front, out)
        };
        let own = |stage: &dyn Scorer| {
            let mut kept = Vec::new();
            let owns = stage.own(
                &facts,
                &PoolFacts::default(),
                &rows,
                &mut |row| row.load.is_some(),
                &mut kept,
            );
            (owns, kept)
        };
        assert_eq!(wrapped.name(), bare.name());
        assert_eq!(score(wrapped.as_ref()), score(&bare));
        assert_eq!(score_front(wrapped.as_ref()), score_front(&bare));
        assert_eq!(wrapped.row_wise(), bare.row_wise());
        assert_eq!(wrapped.front_wise(), bare.front_wise());
        assert_eq!(wrapped.owns(), bare.owns());
        assert_eq!(own(wrapped.as_ref()), own(&bare));
        assert_eq!(wrapped.spread(), bare.spread());
        assert_eq!(wrapped.needs(), bare.needs());
        assert!(std::ptr::addr_eq(
            wrapped.scope().expect("the scope"),
            bare.scope().expect("the scope")
        ));
        assert_eq!(wrapped.refuses_by_reports(), bare.refuses_by_reports());
        assert_eq!(wrapped.warm_up_rule(), bare.warm_up_rule());
        assert_eq!(wrapped.states_overlap(), bare.states_overlap());
        assert_eq!(wrapped.said(&facts, &rows, 1), bare.said(&facts, &rows, 1));
        assert_eq!(wrapped.keeps_loads(), bare.keeps_loads());
        assert_eq!(wrapped.needs_lifecycle(), bare.needs_lifecycle());

        let tell = |stage: &dyn Scorer| {
            let lease = Lease::default();
            let target = target_of(&rows[1]);
            stage.on_decision_issued(&lease, &Decision::Pick(Pick::default()));
            stage.on_dispatched(&lease, &target, Side::Single, 23);
            stage.on_completed(
                &lease,
                &Outcome {
                    side: Side::Single,
                    served: target.id,
                    status: 200,
                    class: TransportClass::Ok,
                    ttft_micros: 0,
                    tokens_in: 0,
                    tokens_out: 0,
                    duration_micros: 0,
                },
            );
            stage.on_released(&lease, ReleaseReason::Abandoned);
            stage.on_membership(&Membership::Added(target.id));
            stage.on_signal(&Signal::Health {
                worker: target.id,
                health: Health::default(),
            });
            std::mem::take(&mut *bare.0.events.lock())
        };
        assert_eq!(tell(wrapped.as_ref()), tell(&bare));
    }

    const COST_TERMS: [(&str, f64); 7] = [
        ("candidate.in_flight", 9.9e14),
        ("load.active_kv_permille", -1.25),
        ("load.kv_usage_permille", 0.75),
        ("load.queued", -9.9e14),
        ("load.resident_tokens", 0.5),
        ("load.running", -0.25),
        ("load.waiting_uncached_tokens", 2.75),
    ];

    fn cost_by_term(terms: &[(CostTerm, f64)], row: &Candidate, price: Option<f64>) -> RowCost {
        let mut sum = 0.0;
        for &(term, coefficient) in terms {
            if let Some(value) = term.read(row.load.as_ref(), row.in_flight) {
                sum += value * coefficient;
            } else if let Some(price) = price {
                return RowCost::OverLeast(f64::from(row.in_flight) * price);
            }
        }
        RowCost::Sum(sum)
    }

    fn cost_bits(cost: RowCost) -> (bool, u64) {
        (
            matches!(cost, RowCost::OverLeast(_)),
            cost.value().to_bits(),
        )
    }

    #[test]
    fn field_sums_match_the_terms_read_in_name_order() {
        let mut draw = Draw::new(17);
        let mut rows: Vec<_> = (0u32..64)
            .map(|position| {
                let mut row = candidate(position, draw.next_u64() as u32);
                row.load = (!position.is_multiple_of(5)).then(|| LoadRow {
                    running: draw.next_u64() as u32,
                    queued: draw.next_u64() as u32,
                    waiting_uncached_tokens: if position.is_multiple_of(2) {
                        0
                    } else {
                        draw.next_u64() as u32
                    },
                    kv_usage_permille: draw.below(1_001) as u16,
                    active_kv_permille: draw.below(1_001) as u16,
                    resident_tokens: draw.next_u64() as u32,
                    resident_counted: position.is_multiple_of(3),
                    ..LoadRow::default()
                });
                row
            })
            .collect();
        // Cancellation makes the smaller terms sensitive to where they were
        // added. An empty report supplies zero waiting tokens.
        rows[0].in_flight = u32::MAX;
        rows[0].load = Some(LoadRow {
            queued: u32::MAX,
            running: 7,
            ..LoadRow::default()
        });
        rows[5] = candidate(5, 0);
        rows[5].load = Some(LoadRow::default());
        let facts = RequestFacts::default();
        let kept = kept(rows.len() as u32);
        let mut costs = Vec::new();
        for mask in 0..1 << COST_TERMS.len() {
            let terms: Box<_> = COST_TERMS
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, &(name, coefficient))| (CostTerm::named(name).unwrap(), coefficient))
                .collect();
            for price in [None, Some(1.5), Some(-0.0)] {
                let rule = price.map_or(Unreported::MissingAsZero, |request_cost| {
                    Unreported::OverLeast { request_cost }
                });
                let stage = LeastCost::new(terms.clone(), CostMargin::default(), Some(rule));
                costs.clear();
                stage.collect_costs(&facts, &rows, &kept, &mut costs, |row| row.load.as_ref());
                assert_eq!(costs.len(), rows.len());
                for (row, &cost) in rows.iter().zip(&costs) {
                    assert_eq!(
                        cost_bits(cost),
                        cost_bits(cost_by_term(&terms, row, price)),
                        "terms {mask:07b}, {rule:?}, row {}",
                        row.position
                    );
                }
            }
        }
    }

    #[test]
    fn a_document_with_one_term_costs_its_field_times_the_coefficient() {
        let mut row = candidate(0, 17);
        row.load = Some(LoadRow {
            active_kv_permille: 19,
            kv_usage_permille: 23,
            queued: 29,
            resident_tokens: 31,
            running: 37,
            waiting_uncached_tokens: 41,
            resident_counted: true,
            ..LoadRow::default()
        });
        let zero = Candidate {
            load: Some(LoadRow {
                resident_counted: true,
                ..LoadRow::default()
            }),
            ..candidate(1, 0)
        };
        let facts = RequestFacts::default();
        let mut costs = Vec::new();
        for ((name, _), value) in COST_TERMS.into_iter().zip([17, 19, 23, 29, 31, 37, 41]) {
            let params: LeastCostParams = serde_yaml::from_str(&format!(
                "terms: {{{name}: -2.5}}\nunreported: missing-as-zero"
            ))
            .unwrap();
            let terms = params
                .terms
                .iter()
                .map(|(name, &coefficient)| (CostTerm::named(name).unwrap(), coefficient))
                .collect();
            let stage = LeastCost::new(terms, params.margin, params.unreported);
            let want = f64::from(value) * -2.5;
            costs.clear();
            stage.collect_costs(
                &facts,
                &[row, zero, candidate(2, 0)],
                &[0, 1, 2],
                &mut costs,
                |row| row.load.as_ref(),
            );
            assert_eq!(cost_bits(costs[0]), (false, want.to_bits()), "{name}");
            assert_eq!(cost_bits(costs[1]), (false, 0.0f64.to_bits()), "{name}");
            assert_eq!(cost_bits(costs[2]), (false, 0.0f64.to_bits()), "{name}");
        }
    }

    #[test]
    fn lent_and_written_rows_score_alike_with_one_lent_reader() {
        let terms = [
            (CostTerm::InFlight, 3.5),
            (CostTerm::named("load.queued").unwrap(), 6.0),
        ];
        let stage = LeastCost::new(
            Box::new(terms),
            CostMargin::default(),
            Some(Unreported::OverLeast { request_cost: 2.5 }),
        );
        let mut rows = [candidate(0, 2), candidate(1, 3), candidate(2, 4)];
        for index in [0, 2] {
            rows[index].load = Some(LoadRow {
                queued: index as u32 + 1,
                ..LoadRow::default()
            });
        }
        let facts = RequestFacts::default();
        let mut written = [0.0; 3];
        stage.score(&facts, &rows, &[0, 1, 2], &mut written);
        let lent = Lent::take(&mut rows);
        let mut scores = [0.0; 3];
        let before = RUN_READERS.get();
        stage.score(&lent.beside(&facts), &rows, &[0, 1, 2], &mut scores);
        assert_eq!(scores.map(f32::to_bits), written.map(f32::to_bits));
        assert_eq!(RUN_READERS.get() - before, 1);
    }

    #[test]
    fn the_sum_and_the_missing_report_price_share_one_count() {
        let stage = LeastCost::new(
            Box::new([
                (CostTerm::InFlight, 2.0),
                (CostTerm::named("load.queued").unwrap(), 3.0),
            ]),
            CostMargin::default(),
            Some(Unreported::OverLeast { request_cost: 4.0 }),
        );
        let count = Cell::new(13);
        let missing = stage.cost_of::<false, false, false>(None, true, || count.replace(29));
        assert_eq!(cost_bits(missing), (true, 52.0f64.to_bits()));
    }

    /// The costs find the same ends by either way: a few rows are compared
    /// one by one, many are folded, and a cost that is no number moves
    /// neither end. So the same costs score the same bits among a few
    /// rows and among many.
    #[test]
    fn a_few_costs_and_many_find_the_same_ends() {
        // The cost that is no number stands last: no cost behind it could
        // take an end back from it.
        let few = [4.0, 2.0, 3.0, f32::NAN];
        let mut some = few;
        let span = normalize_costs(&mut some);
        assert_eq!(span, 2.0);
        assert_eq!(some[0].to_bits(), 0.0f32.to_bits());
        assert_eq!((some[1], some[2]), (1.0, 0.5));
        assert!(some[3].is_nan());
        let mut many: Vec<f32> = few.iter().copied().chain([3.0; FEW_COSTS]).collect();
        assert!(many.len() > FEW_COSTS);
        assert_eq!(normalize_costs(&mut many), span);
        let bits = |scores: &[f32]| scores.iter().map(|s| s.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&many[..few.len()]), bits(&some));
    }

    /// The engine's order around a picker: the set it compares, that set's
    /// scores from `scores_of`, the pick, then the survivors it passed over
    /// as alternates. The positions of the targets come back, in order.
    pub(crate) fn picked(
        picker: &dyn Picker,
        facts: &RequestFacts<'_>,
        pool: &[Candidate],
        scores_of: &dyn Fn(&[u32]) -> Vec<f32>,
    ) -> Vec<u32> {
        picked_within(picker, facts, pool, scores_of, TieBand::NORMALIZED)
    }

    /// [`picked`] under a tie band of the caller's.
    pub(crate) fn picked_within(
        picker: &dyn Picker,
        facts: &RequestFacts<'_>,
        pool: &[Candidate],
        scores_of: &dyn Fn(&[u32]) -> Vec<f32>,
        band: TieBand,
    ) -> Vec<u32> {
        let mut rows = kept(pool.len() as u32);
        let Shortlist {
            compared,
            alternates,
        } = picker.narrow(facts, pool, &mut rows);
        let scores = scores_of(&rows[..compared]);
        let mut out = Targets::new();
        picker.pick(facts, pool, &rows[..compared], &scores, band, &mut out);
        for &row in &rows[compared..compared + alternates] {
            out.push(target_of(&pool[row as usize]));
        }
        out.iter().map(|t| t.position).collect()
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

    pub(crate) fn report(running: u32, waiting_uncached_tokens: u32) -> LoadRow {
        LoadRow {
            running,
            waiting_uncached_tokens,
            live_rate: 2000,
            ..LoadRow::default()
        }
    }

    /// A report that counts resident tokens.
    fn resident(running: u32, resident_tokens: u32) -> LoadRow {
        LoadRow {
            running,
            resident_tokens,
            resident_counted: true,
            ..LoadRow::default()
        }
    }

    /// A report of an engine that does not count them.
    fn uncounted(running: u32) -> LoadRow {
        LoadRow {
            running,
            ..LoadRow::default()
        }
    }

    fn resident_scores<const N: usize>(scorer: &ResidentWork, pool: &[Candidate; N]) -> [f32; N] {
        let mut scores = [0.0f32; N];
        scorer.score(&RequestFacts::default(), pool, &kept(N as u32), &mut scores);
        scores
    }

    /// The request's price is part of the comparison: a worker holding fewer
    /// tokens over more requests loses once a request is priced high enough.
    #[test]
    fn resident_work_is_the_resident_tokens_plus_the_request_price_of_every_request_held() {
        let mut pool = [candidate(0, 2), candidate(1, 1)];
        pool[0].load = Some(resident(2, 6_000));
        pool[1].load = Some(resident(1, 9_000));
        let cheap = ResidentWork::new(1_000);
        assert_eq!(
            cheap.reported_cost(pool[0].in_flight, &resident(2, 6_000), 0, 0),
            8_000.0
        );
        assert_eq!(
            cheap.reported_cost(pool[1].in_flight, &resident(1, 9_000), 0, 0),
            10_000.0
        );
        assert_eq!(resident_scores(&cheap, &pool), [1.0, 0.0]);
        let dear = ResidentWork::new(4_000);
        assert_eq!(
            resident_scores(&dear, &pool),
            [0.0, 1.0],
            "14,000 against 13,000"
        );
        // A queued request is held as a running one is.
        let queued = LoadRow {
            queued: 1,
            ..resident(1, 6_000)
        };
        assert_eq!(
            cheap.reported_cost(pool[0].in_flight, &queued, 0, 0),
            8_000.0
        );
    }

    /// A dispatch the report does not reflect yet adds its prompt and one
    /// request's price; a report sampled after it releases both.
    #[test]
    fn resident_work_credits_a_dispatch_until_a_report_reflects_it() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(resident(1, 4_000));
        pool[1].load_sampled_at_micros = 1_000;
        assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);

        let lease = Lease {
            issued_micros: 2_000,
            tokens: 1_500,
            ..Lease::default()
        };
        rw.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(rw.held_at(pool[0].id, 0), 1);
        assert_eq!(
            resident_scores(&rw, &pool),
            [0.0, 1.0],
            "2,000 + 1,500 + 1,000 for each of two requests, against 5,000"
        );

        pool[0].load_sampled_at_micros = 3_000;
        assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);
        assert_eq!(rw.held_at(pool[0].id, 0), 0);

        // A removed worker leaves no credit behind.
        rw.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        rw.on_membership(&Membership::Removed(pool[1].id));
        assert_eq!(rw.held_at(pool[1].id, 0), 0);
    }

    /// The resident-work stage credits a dispatch of no stated size at its
    /// mean prefill, the crate's default: credited at no tokens, the worker
    /// would still read cheaper than its peer.
    #[test]
    fn resident_work_credits_a_dispatch_of_no_stated_size_at_its_mean_prefill() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(resident(1, 4_000));
        pool[1].load_sampled_at_micros = 1_000;
        assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0]);

        let unstated = Lease {
            issued_micros: 2_000,
            ..Lease::default()
        };
        rw.on_dispatched(&unstated, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(
            resident_scores(&rw, &pool),
            [0.0, 1.0],
            "2,000 + the crate's default 1,024 + 1,000 for each of two requests, against 5,000"
        );
    }

    /// A dispatch of text alone states no size to the resident-work stage
    /// either: it is credited at the stage's mean prefill, and nothing is
    /// read of its text's bytes.
    #[test]
    fn resident_work_credits_a_dispatch_of_text_alone_at_its_mean_prefill() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(resident(1, 4_000));
        pool[1].load_sampled_at_micros = 1_000;
        // 40 bytes of text: 10 tokens by the bytes over four.
        let text = Lease {
            issued_micros: 2_000,
            bytes: 40,
            ..Lease::default()
        };
        rw.on_dispatched(&text, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(
            resident_scores(&rw, &pool),
            [0.0, 1.0],
            "2,000 + the mean's 1,024 + 1,000 for each of two requests, against 5,000"
        );
    }

    /// A report releases the dispatches it reflects whether or not the score
    /// reads their credit. A rank whose report counts no resident tokens
    /// scores from its peers, and holds a dispatch only until a report
    /// sampled after it.
    #[test]
    fn a_report_that_counts_no_resident_tokens_still_releases_its_dispatches() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 1)];
        pool[0].load = Some(uncounted(1));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(resident(1, 4_000));
        pool[1].load_sampled_at_micros = 1_000;
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 1_500,
            ..Lease::default()
        };
        rw.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        let before = resident_scores(&rw, &pool);
        assert_eq!(
            rw.held_at(pool[0].id, 0),
            1,
            "the report predates the dispatch"
        );

        pool[0].load_sampled_at_micros = 3_000;
        assert_eq!(
            resident_scores(&rw, &pool),
            before,
            "the score reads no credit of that rank"
        );
        assert_eq!(rw.held_at(pool[0].id, 0), 0);
    }

    /// A lane's dispatches are held by instant and let go through one: one
    /// dated earlier lands in front, one made at a held instant behind it,
    /// and the oldest goes where it was made at or before the instant.
    #[test]
    fn a_lanes_dispatches_are_held_by_instant_and_let_go_through_one() {
        let mut dispatches = Dispatches::default();
        assert_eq!(dispatches.oldest(), None);
        // The first held is the oldest. A later one is not.
        assert!(dispatches.hold(20, SendSize::stated(2)));
        assert!(!dispatches.hold(30, SendSize::stated(3)));
        // One dated earlier is the oldest now.
        assert!(dispatches.hold(10, SendSize::stated(1)));
        // One dated between lands between, one at a held instant behind it.
        assert!(!dispatches.hold(25, SendSize::stated(4)));
        assert!(!dispatches.hold(20, SendSize::stated(5)));
        assert_eq!((dispatches.oldest(), dispatches.len()), (Some(10), 5));
        // The oldest stays while the instant is before it and goes at its
        // own instant. The next stays one microsecond short of its own.
        assert_eq!(dispatches.let_go(9), None);
        assert_eq!(dispatches.let_go(10), Some(SendSize::stated(1)));
        assert_eq!(dispatches.let_go(19), None);
        assert_eq!(dispatches.oldest(), Some(20));
        // Oldest first, and of one instant the earlier held first.
        let gone: Vec<u32> = std::iter::from_fn(|| dispatches.let_go(u64::MAX))
            .map(SendSize::tokens)
            .collect();
        assert_eq!(gone, [2, 5, 4, 3]);
        assert_eq!((dispatches.oldest(), dispatches.len()), (None, 0));
    }

    /// A credit is its rank's: a dispatch to rank 1 of a worker prices that
    /// rank's row and leaves rank 0's alone.
    #[test]
    fn a_dispatch_is_credited_to_the_rank_it_went_to() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[1].id = pool[0].id;
        pool[1].dp_rank = 1;
        for c in &mut pool {
            c.load = Some(resident(1, 2_000));
            c.load_sampled_at_micros = 1_000;
        }
        assert_eq!(resident_scores(&rw, &pool), [1.0, 1.0]);

        let lease = Lease {
            issued_micros: 2_000,
            tokens: 1_500,
            ..Lease::default()
        };
        rw.on_dispatched(&lease, &target_of(&pool[1]), Side::Single, 0);
        assert_eq!(
            (rw.held_at(pool[0].id, 0), rw.held_at(pool[0].id, 1)),
            (0, 1)
        );
        assert_eq!(
            resident_scores(&rw, &pool),
            [1.0, 0.0],
            "3,000 on rank 0 against 2,000 + 1,500 + 1,000 for each of two requests on rank 1"
        );
    }

    /// The mean context is the resident tokens over the requests the report
    /// holds, the queued with the running: 8,000 tokens over two running and
    /// two queued price the one request in transit at 2,000.
    #[test]
    fn the_mean_context_counts_the_queued_requests_with_the_running() {
        let rw = ResidentWork::new(1_000);
        let busy = candidate(0, 5);
        let row = LoadRow {
            queued: 2,
            ..resident(2, 8_000)
        };
        assert_eq!(
            rw.reported_cost(busy.in_flight, &row, 0, 0),
            8_000.0 + 2_000.0 + 5.0 * 1_000.0
        );
    }

    /// The band is the scorer's own, on its score in tokens, and not the
    /// plain band of a normalized score: with one worker two million tokens
    /// behind, two workers a token apart are still not a tie.
    #[test]
    fn a_token_apart_is_not_a_tie_under_a_wide_spread() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        pool[0].load = Some(resident(1, 5_000));
        pool[1].load = Some(resident(1, 5_001));
        pool[2].load = Some(resident(1, 2_005_000));
        let mut scores = [0.0f32; 3];
        let band = rw.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert!(scores[0] - scores[1] > band.0, "{scores:?} {band:?}");
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            assert_eq!(out.first().map(|t| t.position), Some(0), "seed {seed}");
        }
    }

    /// The requests the host holds beyond the report and the credit are in
    /// transit: each is priced at the worker's mean context and at the
    /// request price; a worker that reports nothing held prices them at the
    /// mean prefill.
    #[test]
    fn resident_work_prices_the_requests_in_transit_at_the_workers_mean_context() {
        let rw = ResidentWork::new(1_000);
        let busy = candidate(0, 5);
        // Two reported requests hold 8,000 tokens, a mean context of 4,000;
        // three more are in transit.
        assert_eq!(
            rw.reported_cost(busy.in_flight, &resident(2, 8_000), 0, 0),
            8_000.0 + 3.0 * 4_000.0 + 5.0 * 1_000.0
        );
        // One of the three is a credited dispatch of 1,500 tokens.
        assert_eq!(
            rw.reported_cost(busy.in_flight, &resident(2, 8_000), 1_500, 1),
            8_000.0 + 1_500.0 + 2.0 * 4_000.0 + 5.0 * 1_000.0
        );
        let idle = candidate(1, 2);
        assert_eq!(
            rw.reported_cost(idle.in_flight, &resident(0, 0), 0, 0),
            2.0 * DEFAULT_MEAN_PREFILL_TOKENS as f32 + 2.0 * 1_000.0
        );
    }

    /// A worker whose report does not count resident tokens, or that has no
    /// report, starts from the best reporting peer and adds its in-flight
    /// requests, each at the mean prefill and the request price.
    #[test]
    fn a_worker_without_resident_tokens_scores_from_the_best_reporting_peer() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 2), candidate(2, 0)];
        pool[0].load = Some(resident(1, 5_000));
        pool[1].load = Some(uncounted(2));
        // 6,000; 6,000 + 2 * 2,024; and the unreported idle worker at 6,000.
        assert_eq!(resident_scores(&rw, &pool), [1.0, 0.0, 1.0]);
    }

    /// The peer it starts from is the best one, not the worst, and each of
    /// its requests in flight costs the mean prefill and the request price:
    /// between a peer at 6,000 and a peer at 14,096, two requests in flight
    /// put it at 6,000 + 2 * 2,024, the exact middle.
    #[test]
    fn a_worker_without_resident_tokens_is_priced_between_two_reporting_peers() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 2), candidate(2, 1)];
        pool[0].load = Some(resident(1, 5_000));
        pool[1].load = Some(uncounted(2));
        pool[2].load = Some(resident(1, 13_096));
        assert_eq!(resident_scores(&rw, &pool), [1.0, 0.5, 0.0]);
    }

    #[test]
    fn with_no_survivor_reporting_resident_tokens_the_in_flight_count_decides() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 3), candidate(1, 1), candidate(2, 2)];
        pool[0].load = Some(uncounted(3));
        assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0, 0.5]);
    }

    /// In a fleet whose engines do not count resident tokens, a report that
    /// reads idle is no reporting peer: the in-flight count decides, and the
    /// worker the host holds thirty requests on does not take the next one.
    #[test]
    fn an_idle_report_without_the_count_makes_no_reporting_peer() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 30), candidate(1, 1)];
        pool[0].load = Some(uncounted(0));
        pool[1].load = Some(uncounted(1));
        assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
    }

    /// A count of zero and no count are two reports. The idle worker of an
    /// engine that counts resident tokens scores least, and so does one
    /// whose running request has nothing resident yet; the idle worker of an
    /// engine that does not count them ties with the best reporting peer.
    #[test]
    fn a_count_of_zero_is_not_a_report_without_the_count() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 0)];
        pool[0].load = Some(resident(1, 5_000));
        pool[1].load = Some(resident(0, 0));
        assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
        pool[1].load = Some(uncounted(0));
        assert_eq!(resident_scores(&rw, &pool), [1.0, 1.0]);
        pool[1] = candidate(1, 1);
        pool[1].load = Some(resident(1, 0));
        assert_eq!(resident_scores(&rw, &pool), [0.0, 1.0]);
    }

    /// Equal work is one tie, and a token apart is not: the band is the
    /// least-load scorer's, on a score in tokens.
    #[test]
    fn equal_resident_work_ties_and_a_token_apart_does_not() {
        let rw = ResidentWork::new(1_000);
        let mut pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        pool[0].load = Some(resident(1, 5_000));
        pool[1].load = Some(resident(1, 5_000));
        pool[2].load = Some(resident(1, 5_001));
        let mut scores = [0.0f32; 3];
        let band = rw.score(&RequestFacts::default(), &pool, &kept(3), &mut scores);
        assert_eq!(scores, [1.0, 1.0, 0.0]);
        assert!(band.0 < 1.0, "{band:?}");
        let mut picks = [0usize; 3];
        for seed in 0..300u64 {
            let facts = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let mut out = Targets::new();
            Best.pick(&facts, &pool, &kept(3), &scores, band, &mut out);
            picks[out.first().map(|t| t.position).unwrap_or(0) as usize] += 1;
        }
        assert!(picks[0] > 100 && picks[1] > 100, "{picks:?}");
        assert_eq!(picks[2], 0, "{picks:?}");
    }

    #[test]
    fn a_read_across_a_release_counts_no_more_unsized_dispatches_than_dispatches() {
        // One thread releases a row that holds dispatches of no tokens
        // alone, and this one reads the row's two words as they stand. The
        // count of those dispatches goes down ahead of the held word at
        // each one, so no read sees more of them than dispatches. A round
        // whose reads all land beside the release shows nothing: the
        // rounds go on until a read lands inside one, a few at most.
        const HELD: u64 = 6_000;
        for _ in 0..32 {
            let row = RankCredit::default();
            for at in 1..=HELD {
                row.record(at, SendSize::stated(0));
            }
            let crossed = std::thread::scope(|s| {
                let release = s.spawn(|| row.sent_through(HELD));
                let mut crossed = false;
                while !release.is_finished() {
                    // An instant ahead of every dispatch releases none.
                    let read = row.sent_through(0);
                    assert!(
                        read.unsized_requests <= read.requests,
                        "{} dispatches of no tokens beside {} dispatches",
                        read.unsized_requests,
                        read.requests
                    );
                    crossed |= read.requests != 0 && u64::from(read.requests) != HELD;
                }
                crossed
            });
            if crossed {
                return;
            }
        }
    }

    /// The estimates' word goes down ahead of the held word at each
    /// dispatch a release lets go, so a read of the documents' two numbers
    /// that crosses the release adds no estimate of a dispatch the held
    /// word has let go. Every dispatch here is an estimate of three
    /// tokens: no read finds more than three tokens a dispatch. The rounds
    /// go on until a read lands inside a release, a few at most.
    #[test]
    fn a_read_across_a_release_adds_no_estimate_of_a_dispatch_let_go() {
        const HELD: u64 = 6_000;
        const EACH: u32 = 3;
        for _ in 0..32 {
            let row = RankCredit::default();
            for at in 1..=HELD {
                row.record(at, SendSize::estimated(EACH));
            }
            let crossed = std::thread::scope(|s| {
                let release = s.spawn(|| row.counts_through(HELD));
                let mut crossed = false;
                while !release.is_finished() {
                    // An instant ahead of every dispatch releases none.
                    let read = row.counts_through(0);
                    assert!(
                        read.tokens <= u64::from(EACH) * u64::from(read.requests),
                        "{} estimated tokens beside {} dispatches",
                        read.tokens,
                        read.requests
                    );
                    crossed |= read.requests != 0 && u64::from(read.requests) != HELD;
                }
                crossed
            });
            if crossed {
                return;
            }
        }
    }

    /// The estimates' word goes down ahead of the held word at a
    /// retirement too. Every dispatch here is an estimate of three tokens,
    /// and the dispatch past the row's bound lets the row's oldest go. A
    /// read of the documents' two numbers between the retirement's two
    /// writes finds the held word still counting that one and no more than
    /// three tokens a dispatch: it adds no estimate of a dispatch the held
    /// word has let go.
    #[test]
    fn a_read_between_a_retirements_writes_adds_no_estimate_of_a_dispatch_let_go() {
        const EACH: u32 = 3;
        thread_local!(static READS: Cell<u32> = const { Cell::new(0) });
        fn read(row: &RankCredit) {
            // An instant ahead of every dispatch releases none.
            let read = row.counts_through(0);
            assert!(
                read.tokens <= u64::from(EACH) * u64::from(read.requests),
                "{} estimated tokens beside {} dispatches",
                read.tokens,
                read.requests
            );
            READS.set(READS.get() + 1);
        }
        let row = RankCredit::default();
        for at in 1..=RANK_CREDIT_KEPT {
            row.record(at, SendSize::estimated(EACH));
        }
        BETWEEN_A_RETIREMENTS_WRITES.set(Some(read));
        row.record(RANK_CREDIT_KEPT + 1, SendSize::estimated(EACH));
        BETWEEN_A_RETIREMENTS_WRITES.set(None);
        assert_eq!(READS.get(), 1, "one retirement, read between its writes");
        let kept = RANK_CREDIT_KEPT;
        let words = (0, kept, kept as u16, u64::from(EACH) * kept);
        assert_eq!(row.words_for_test(), words);
    }

    /// A dispatch that states no tokens states no size: it is held as an
    /// estimate of none, among the sends of no stated size.
    #[test]
    fn a_stated_size_of_no_tokens_is_held_as_a_send_of_no_stated_size() {
        assert_eq!(SendSize::stated(0), SendSize::estimated(0));
        let row = RankCredit::default();
        row.record(1, SendSize::stated(0));
        assert_eq!(row.words_for_test(), (0, 1, 1, 0));
    }

    /// A size is held under the mark of an estimate, stated or estimated:
    /// one at the mark's bit and one past it are held as the largest
    /// under it. A stated one stays a stated one. It is not read as an
    /// estimate, and its release takes it out of the held word and
    /// leaves the estimates' word as it stands.
    #[test]
    fn a_size_at_the_marks_bit_is_held_under_it_and_in_its_own_word() {
        const MOST: u64 = (1 << 31) - 1;
        for size in [1u32 << 31, u32::MAX] {
            let row = RankCredit::default();
            row.record(1, SendSize::stated(size));
            assert_eq!(row.words_for_test(), (MOST, 1, 0, 0), "stated {size}");
            row.record(2, SendSize::estimated(size));
            assert_eq!(row.words_for_test(), (MOST, 2, 1, MOST), "{size}");
            // A report sampled at the first lets the stated one go alone.
            let read = row.counts_through(1);
            assert_eq!((read.requests, read.tokens), (1, MOST), "{size}");
            assert_eq!(row.words_for_test(), (0, 1, 1, MOST), "{size}");
            row.counts_through(2);
            assert_eq!(row.words_for_test(), (0, 0, 0, 0), "{size}");
        }
    }

    /// A lane that holds a stated size, an estimate and a stated size
    /// reads back to nothing in each of the row's words, by a release and
    /// by a retirement: an estimate goes with its dispatch. The
    /// documents' read adds the estimate to the stated token-work, and
    /// the read of a stage that prices such a send itself counts it and
    /// adds none.
    #[test]
    fn a_mixed_lane_reads_back_to_nothing_by_a_release_and_by_a_retirement() {
        let mixed = [
            SendSize::stated(7),
            SendSize::estimated(101),
            SendSize::stated(5),
        ];
        let row = RankCredit::default();
        for (at, size) in (1..).zip(mixed) {
            row.record(at, size);
        }
        assert_eq!(row.words_for_test(), (12, 3, 1, 101));
        // An instant ahead of every dispatch releases none.
        let read = row.counts_through(0);
        assert_eq!((read.requests, read.tokens), (3, 113));
        let sent = row.sent_through(0);
        assert_eq!((sent.requests, sent.tokens), (3, 12));
        assert_eq!(sent.unsized_requests, 1);
        // A release, one dispatch at a time.
        row.counts_through(1);
        assert_eq!(row.words_for_test(), (5, 2, 1, 101));
        row.counts_through(2);
        assert_eq!(row.words_for_test(), (5, 1, 0, 0));
        row.counts_through(3);
        assert_eq!(row.words_for_test(), (0, 0, 0, 0));

        // A retirement: each of two dispatches that land past the row's
        // bound lets the row's oldest go, a stated size and then an
        // estimate.
        let row = RankCredit::default();
        for at in 0..RANK_CREDIT_KEPT {
            row.record(at + 1, mixed[(at % 3) as usize]);
        }
        assert_eq!(
            row.words_for_test(),
            (2_731 * 7 + 2_730 * 5, 8_192, 2_731, 2_731 * 101)
        );
        for at in RANK_CREDIT_KEPT..RANK_CREDIT_KEPT + 2 {
            row.record(at + 1, mixed[(at % 3) as usize]);
        }
        assert_eq!(
            row.words_for_test(),
            (2_731 * 12, 8_192, 2_730, 2_730 * 101),
            "the oldest two went, each with its own word's share"
        );
        row.counts_through(u64::MAX);
        assert_eq!(row.words_for_test(), (0, 0, 0, 0));
    }

    /// A read of a row's dispatches whose instant is short of the row's
    /// oldest writes nothing of the row. A retirement leaves that instant
    /// where it was, below the oldest dispatch the row still holds: a read
    /// short of it hands every held dispatch and leaves it there. A read
    /// that reaches it runs the release, which lets nothing go here and
    /// states the oldest held dispatch's instant.
    #[test]
    fn a_read_of_a_rows_dispatches_short_of_its_oldest_instant_writes_nothing_of_the_row() {
        let row = RankCredit::default();
        // The dispatch that finds the row at its bound lets the oldest go.
        let (first, last) = (10, 10 + RANK_CREDIT_KEPT);
        for at in first..=last {
            row.record(at, SendSize::stated(1));
        }
        let oldest_held = first + 1;
        let held: Vec<u64> = (oldest_held..=last).collect();
        let oldest = || row.oldest_micros.load(Ordering::Acquire);
        let handed = |through_micros| {
            let mut handed = Vec::new();
            row.each_through(through_micros, |at_micros, _| handed.push(at_micros));
            handed
        };
        assert_eq!(oldest(), first, "a retirement leaves it");
        assert_eq!(handed(first - 1), held);
        assert_eq!(oldest(), first, "a read short of it stores none");
        assert_eq!(handed(first), held);
        assert_eq!(oldest(), oldest_held, "a read that reaches it releases");
    }

    #[test]
    fn the_session_scorer_reads_the_pin_beside_the_rows_own_bit() {
        let pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let unpinned = RequestFacts::default();
        let pinned = RequestFacts {
            pinned_at: 2,
            ..unpinned
        };
        assert!(pinned.pins(1) && !pinned.pins(0) && !pinned.pins(2));
        assert!(!unpinned.pins(0) && !unpinned.pins(u32::MAX - 1));
        // No position reads as pinned under no pin, the last one neither.
        assert!(!unpinned.pins(u32::MAX));
        let mut scores = [0.0f32; 3];
        SessionAffinity.score(&unpinned, &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 0.0, 0.0]);
        SessionAffinity.score(&pinned, &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 1.0, 0.0]);
        // The scorer is handed the rows kept: the pin is read by position.
        let mut scores = [0.0f32; 2];
        SessionAffinity.score(&pinned, &pool, &[1, 2], &mut scores);
        assert_eq!(scores, [1.0, 0.0]);
    }

    /// Every built-in that reads a row's requests in flight reads them
    /// through the request's facts: over rows whose counts stand in a lent
    /// table, the rows themselves reading zero, each scores and picks as it
    /// does over rows that carry the counts.
    #[test]
    fn a_built_in_reads_a_lent_count_as_it_reads_a_rows_own() {
        // Counts that part the rows, and reports on four of them: both
        // load scorers take their reporting and their unreporting terms,
        // and two rows that count resident tokens part by their costs.
        let counts = [5, 1, 3, 0, 4, 2, 6, 1];
        let mut pool: Vec<Candidate> = (0..)
            .zip(counts)
            .map(|(at, held)| candidate(at, held))
            .collect();
        pool[0].load = Some(resident(2, 9_000));
        pool[1].load = Some(report(2, 4_000));
        pool[4].load = Some(resident(1, 6_000));
        pool[6].load = Some(report(0, 0));
        let mut bare = pool.clone();
        let lent = Lent::take(&mut bare);
        assert!(bare.iter().all(|row| row.in_flight == 0));
        let rows = kept(pool.len() as u32);
        let flat = |rows: &[u32]| vec![0.0; rows.len()];
        for seed in 0..16 {
            let own = RequestFacts {
                seed,
                ..RequestFacts::default()
            };
            let beside = lent.beside(&own);
            let scorers: [&dyn Scorer; 3] = [
                &LeastRequests,
                &LeastLoad::new(0.15, 1024, 2000.0),
                &ResidentWork::new(512),
            ];
            for scorer in scorers {
                let (mut written, mut read) = (vec![0.0; pool.len()], vec![0.0; pool.len()]);
                scorer.score(&own, &pool, &rows, &mut written);
                scorer.score(&beside, &bare, &rows, &mut read);
                assert_eq!(written, read, "{} at seed {seed}", scorer.name());
            }
            let pickers: [&dyn Picker; 2] = [&CountFirst, &PowerOfTwoCountFirst];
            for picker in pickers {
                let written = picked(picker, &own, &pool, &flat);
                let read = picked(picker, &beside, &bare, &flat);
                assert_eq!(written, read, "{} at seed {seed}", picker.name());
            }
        }
    }

    /// The number a stage keeps of `row`, where it keeps the row's report.
    fn kept_of(stage: &ResidentWork, slot: WorkerSlot, row: &Candidate) -> Option<(f32, u32)> {
        stage.kept.cell(slot).and_then(|cell| cell.read(row))
    }

    /// Tells `stage` the report `row` holds, as a host tells it.
    fn tell(stage: &ResidentWork, slot: WorkerSlot, row: &Candidate) {
        stage.on_signal(&Signal::Load {
            worker: row.id,
            dp_rank: row.dp_rank,
            row: row.load.expect("a report to tell"),
            slot,
            sampled_at_micros: row.load_sampled_at_micros,
            signals: ReportSignals::NONE,
        });
    }

    /// The first decision under the report `row` holds, beside `lent`: the
    /// row costed alone, as a draw of one row costs it. A stage told the
    /// report keeps its number from this decision on.
    fn decide<'a>(stage: &ResidentWork, lent: &'a Lent<'a>, row: &Candidate) {
        let facts = lent.beside(&RequestFacts::default());
        stage.score(&facts, std::slice::from_ref(row), &[0], &mut [0.0]);
    }

    /// Tells `stage` the report `row` holds and decides under it once, as
    /// a host shows the world it told.
    fn shown<'a>(stage: &ResidentWork, lent: &'a Lent<'a>, slot: WorkerSlot, row: &Candidate) {
        tell(stage, slot, row);
        decide(stage, lent, row);
    }

    /// A cell keeps a number under its report's stamp and hands it to a row
    /// that holds that very report: no number under another stamp, to a row
    /// with no report, before a write and after a clear.
    #[test]
    fn a_kept_number_is_read_under_its_reports_stamp_alone() {
        let cell = Kept::default();
        let mut row = candidate(0, 0);
        row.load = Some(resident(1, 2_000));
        row.load_sampled_at_micros = 1_000;
        assert_eq!(cell.read(&row), None, "an unwritten cell");
        cell.write(1_000, 3_500.0, 4);
        assert_eq!(cell.read(&row), Some((3_500.0, 4)));
        row.load_sampled_at_micros = 1_001;
        assert_eq!(cell.read(&row), None, "another report's stamp");
        row.load_sampled_at_micros = 1_000;
        row.load = None;
        assert_eq!(cell.read(&row), None, "a row with no report");
        row.load = Some(resident(1, 2_000));
        // More requests than the half-word holds are every request.
        cell.write(1_000, 1.0, u64::from(u32::MAX) + 9);
        assert_eq!(cell.read(&row), Some((1.0, u32::MAX)));
        // A read inside a write is handed nothing, whatever stands.
        cell.turn.fetch_add(1, Ordering::Relaxed);
        assert_eq!(cell.read(&row), None, "a write at the cell");
        cell.turn.fetch_add(1, Ordering::Relaxed);
        assert_eq!(cell.read(&row), Some((1.0, u32::MAX)));
        cell.clear();
        assert_eq!(cell.read(&row), None, "a cleared cell");
        // No report is stamped zero: a row that says zero is kept nothing.
        cell.write(0, 1.0, 1);
        row.load_sampled_at_micros = 0;
        assert_eq!(cell.read(&row), None, "the stamp of no report");
    }

    /// A read that a write crosses is handed no number of one report under
    /// another's stamp. One writer turns a cell between two reports, each
    /// with a number of its own, and clears it between them, while two
    /// readers ask for each report: every number a reader is handed is its
    /// report's.
    #[test]
    fn a_read_a_write_crosses_pairs_no_number_with_another_reports_stamp() {
        const TURNS: u64 = 200_000;
        let cell = Kept::default();
        let rows = [1_000u64, 2_000].map(|stamp| {
            let mut row = candidate(0, 0);
            row.load = Some(resident(1, 1));
            row.load_sampled_at_micros = stamp;
            row
        });
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| {
            for _ in 0..2 {
                scope.spawn(|| {
                    while !done.load(Ordering::Relaxed) {
                        for row in &rows {
                            // A report's number is its own stamp, twice.
                            let stamp = row.load_sampled_at_micros;
                            if let Some((cost, accounted)) = cell.read(row) {
                                assert_eq!((cost, u64::from(accounted)), (stamp as f32, stamp));
                            }
                        }
                    }
                });
            }
            for turn in 0..TURNS {
                let stamp = rows[(turn % 2) as usize].load_sampled_at_micros;
                cell.write(stamp, stamp as f32, stamp);
                if turn % 3 == 0 {
                    cell.clear();
                }
            }
            done.store(true, Ordering::Relaxed);
        });
    }

    /// An entry the tally let go names no cell: a dispatch that found it
    /// before its rank left writes no number behind the leave. The rank's
    /// next report finds a new entry, and is kept once its instant is past
    /// the last the leaver was told.
    #[test]
    fn an_entry_the_tally_let_go_writes_no_number() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        let lent = Lent::take(&mut pool);
        let slot = lent.slot_of(&pool[0]);
        tell(&stage, slot, &pool[0]);
        assert_eq!(kept_of(&stage, slot, &pool[0]), None, "ahead of a decision");
        decide(&stage, &lent, &pool[0]);
        assert_eq!(kept_of(&stage, slot, &pool[0]), Some((2_700.0, 1)));
        let held = stage.credits.entry((pool[0].id, 0)).expect("the entry");
        stage.on_membership(&Membership::Removed(pool[0].id));
        assert_eq!(kept_of(&stage, slot, &pool[0]), None, "the leaver's");
        assert_eq!(*held.told.lock(), Told::default(), "it names a cell");

        shown(&stage, &lent, slot, &pool[0]);
        let first = kept_of(&stage, slot, &pool[0]);
        assert_eq!(first, None, "an instant the leaver was told");
        let mut newer = pool[0];
        newer.load_sampled_at_micros = 2_000;
        shown(&stage, &lent, slot, &newer);
        assert_eq!(kept_of(&stage, slot, &newer), Some((2_700.0, 1)));
    }

    /// A member whose rank a registration moved keeps its slot, and the
    /// entry of the rank it was is credited still: decisions made before
    /// the move name it. Once the rank it is has been told, that entry
    /// writes the member's cell no more.
    #[test]
    fn an_entry_writes_no_number_once_its_cell_is_told_under_another_rank() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        let lent = Lent::take(&mut pool);
        let slot = lent.slot_of(&pool[0]);
        let was = pool[0];
        shown(&stage, &lent, slot, &was);
        assert_eq!(kept_of(&stage, slot, &was), Some((2_700.0, 1)));

        let mut is = was;
        is.dp_rank = 1;
        is.load = Some(resident(2, 4_000));
        is.load_sampled_at_micros = 2_000;
        shown(&stage, &lent, slot, &is);
        assert_eq!(kept_of(&stage, slot, &is), Some((5_400.0, 2)));

        let lease = Lease {
            issued_micros: 3_000,
            tokens: 1_500,
            ..Lease::default()
        };
        stage.on_dispatched(&lease, &target_of(&was), Side::Single, 0);
        let stands = kept_of(&stage, slot, &is);
        assert_eq!(stands, Some((5_400.0, 2)), "the other rank's entry wrote");
        assert_eq!(kept_of(&stage, slot, &was), None);
    }

    /// A rank told under a second slot lets the cell of its first go: a
    /// row that still holds the first slot and the report told there is
    /// costed from its parts, behind a dispatch too. The cell keeps no
    /// number of that instant for another rank told under it, and keeps
    /// the next instant's.
    #[test]
    fn a_rank_told_under_a_second_slot_lets_its_first_cell_go() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        let lent = Lent::take(&mut pool);
        let (first, second) = (lent.slot_of(&pool[0]), lent.slot_of(&pool[1]));
        let was = pool[0];
        shown(&stage, &lent, first, &was);
        assert_eq!(kept_of(&stage, first, &was), Some((2_700.0, 1)));

        // The rank's row as the next world shows it, at the second
        // slot's place.
        let mut is = Candidate {
            position: pool[1].position,
            ..was
        };
        is.load = Some(resident(2, 4_000));
        is.load_sampled_at_micros = 2_000;
        shown(&stage, &lent, second, &is);
        assert_eq!(kept_of(&stage, second, &is), Some((5_400.0, 2)));
        assert_eq!(kept_of(&stage, first, &was), None, "the cell let go");

        let lease = Lease {
            issued_micros: 3_000,
            tokens: 1_500,
            ..Lease::default()
        };
        stage.on_dispatched(&lease, &target_of(&is), Side::Single, 0);
        assert_eq!(kept_of(&stage, first, &was), None, "behind a dispatch");
        assert_eq!(kept_of(&stage, second, &is), Some((7_600.0, 3)));

        // Another rank told under the first slot at that instant is kept
        // no number of it: the row above names that instant too.
        let mut other = Candidate {
            position: was.position,
            ..candidate(2, 0)
        };
        other.load = Some(resident(3, 6_000));
        other.load_sampled_at_micros = 1_000;
        shown(&stage, &lent, first, &other);
        assert_eq!(kept_of(&stage, first, &other), None, "that instant");
        assert_eq!(kept_of(&stage, first, &was), None, "another rank's");
        other.load_sampled_at_micros = 1_500;
        shown(&stage, &lent, first, &other);
        assert_eq!(kept_of(&stage, first, &other), Some((8_100.0, 3)));
    }

    /// A cell that changes hands keeps no number of an instant the entry
    /// that wrote it before was told a report of: a decision that still
    /// reads that entry's row would be handed another rank's number under
    /// its own row's stamp. A rank moved under its slot, and a rank told
    /// again after a resize let its entry go, keep none of the last
    /// instant told before them, and the next instant's number.
    #[test]
    fn a_cell_keeps_no_number_of_an_instant_its_entry_of_before_was_told() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        let lent = Lent::take(&mut pool);
        let slot = lent.slot_of(&pool[0]);
        let was = pool[0];
        shown(&stage, &lent, slot, &was);
        assert_eq!(kept_of(&stage, slot, &was), Some((2_700.0, 1)));

        let mut is = was;
        is.dp_rank = 1;
        is.load = Some(resident(2, 4_000));
        shown(&stage, &lent, slot, &is);
        assert_eq!(
            kept_of(&stage, slot, &is),
            None,
            "the instant of the rank it was"
        );
        assert_eq!(kept_of(&stage, slot, &was), None, "the rank it was");
        is.load_sampled_at_micros = 2_000;
        shown(&stage, &lent, slot, &is);
        assert_eq!(kept_of(&stage, slot, &is), Some((5_400.0, 2)));

        stage.on_membership(&Membership::Resized(is.id));
        assert_eq!(kept_of(&stage, slot, &is), None, "the resized member's");
        let mut back = was;
        back.load_sampled_at_micros = 2_000;
        shown(&stage, &lent, slot, &back);
        assert_eq!(
            kept_of(&stage, slot, &back),
            None,
            "the instant of the entry let go"
        );
        back.load_sampled_at_micros = 3_000;
        shown(&stage, &lent, slot, &back);
        assert_eq!(kept_of(&stage, slot, &back), Some((2_700.0, 1)));
    }

    /// One entry writes a cell at a time, whatever ranks its slot is told
    /// under: two ranks of one member are told in turn under one slot,
    /// each instant once a rank and the first of the two by turns, while a
    /// thread a rank credits dispatches and two more decide and read. A
    /// row is decided once its report was told, as a host shows a world it
    /// told, and the row told before it is decided on: every number a row
    /// is handed is its own rank's. The round ends on one more instant, of
    /// the rank told last alone, so a reader is handed a number on a
    /// runner of any size. On one or two cores that number may be the
    /// only one: the turns then showed no hand-over of their own.
    #[test]
    fn a_slot_told_under_two_ranks_hands_each_row_its_own_ranks_number() {
        const INSTANTS: u64 = 20_000;
        // A rank's reports count resident tokens in a band of its own.
        const BANDS: [u32; 2] = [1_000_000, 1_000_000_000];
        let stage = ResidentWork::new(1);
        let mut pool = [candidate(0, 0)];
        let lent = Lent::take(&mut pool);
        let slot = lent.slot_of(&pool[0]);
        // The row a rank shows at an instant, by the word that names both.
        let row_of = |shown: u64| {
            let mut row = pool[0];
            row.dp_rank = (shown & 1) as u16;
            row.load = Some(resident(0, BANDS[usize::from(row.dp_rank)]));
            row.load_sampled_at_micros = shown >> 1;
            row
        };
        let (shown, done) = (AtomicU64::new(0), AtomicBool::new(false));
        let handed = AtomicU64::new(0);
        std::thread::scope(|scope| {
            for rank in 0..2 {
                scope.spawn(|| {
                    // A table of the reader's own: the one row books the
                    // same slot in each.
                    let mut own = [candidate(0, 0)];
                    let lent = Lent::take(&mut own);
                    assert_eq!(lent.slot_of(&own[0]), slot);
                    let mut before = 0;
                    while !done.load(Ordering::Relaxed) {
                        let last = shown.load(Ordering::Acquire);
                        for word in [last, before] {
                            let row = row_of(word);
                            // A number is kept from a report's first
                            // decision on.
                            decide(&stage, &lent, &row);
                            if let Some((cost, _)) = kept_of(&stage, slot, &row) {
                                let band = usize::from(cost >= 500_000_000.0);
                                assert_eq!(band, usize::from(row.dp_rank), "at {word}");
                                handed.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                        before = last;
                    }
                });
                let (stage, shown, done) = (&stage, &shown, &done);
                scope.spawn(move || {
                    let target = target_of(&row_of(rank));
                    while !done.load(Ordering::Relaxed) {
                        let lease = Lease {
                            issued_micros: shown.load(Ordering::Relaxed) >> 1,
                            tokens: 1,
                            ..Lease::default()
                        };
                        stage.on_dispatched(&lease, &target, Side::Single, 0);
                    }
                });
            }
            for instant in 1..=INSTANTS {
                for turn in 0..2 {
                    let word = instant << 1 | ((instant + turn) & 1);
                    tell(&stage, slot, &row_of(word));
                    shown.store(word, Ordering::Release);
                }
            }
            // One more word, of the rank told last. A cell keeps no number
            // of an instant it was told two ranks under, so the last word
            // above hands over none, and this one can. The round ends once
            // a reader was handed a number: on one or two cores the tells
            // could end before any reader ran between two of them.
            let last = (INSTANTS + 1) << 1 | ((INSTANTS + 1) & 1);
            tell(&stage, slot, &row_of(last));
            shown.store(last, Ordering::Release);
            let patience = std::time::Instant::now();
            while handed.load(Ordering::Relaxed) == 0
                && patience.elapsed() < std::time::Duration::from_secs(30)
            {
                std::thread::yield_now();
            }
            done.store(true, Ordering::Relaxed);
        });
        let handed = handed.load(Ordering::Relaxed);
        assert!(handed > 0, "no reader was handed a number");
    }

    /// A stage told its ranks' reports scores every row as a stage told
    /// nothing does, bit for bit, at each step of a rank's life: its report
    /// told, a dispatch credited, a newer report shown and then told, a
    /// late one, another report under the same instant, a report that
    /// counts no resident tokens, and the rank's leave. It keeps a number
    /// of each row whose report counts resident tokens, the row that holds
    /// one request more than that number accounts for among them: that row
    /// is costed from its parts.
    #[test]
    fn a_told_stage_scores_its_rows_as_an_untold_one_does() {
        let mut pool = [
            candidate(0, 1),
            candidate(1, 2),
            candidate(2, 3),
            candidate(3, 1),
        ];
        pool[0].load = Some(resident(1, 2_000));
        pool[1].load = Some(resident(2, 4_000));
        pool[2].load = Some(resident(2, 3_000));
        pool[3].load = Some(uncounted(1));
        for row in &mut pool {
            row.load_sampled_at_micros = 1_000;
        }
        let lent = Lent::take(&mut pool);
        let slots = [0, 1, 2, 3].map(|at: usize| lent.slot_of(&pool[at]));
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let scores = |stage: &ResidentWork, pool: &[Candidate; 4]| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 4];
            stage.score(&facts, pool, &kept(4), &mut out);
            out.map(f32::to_bits)
        };
        // A world's first read is its reports' first decision, and the
        // read behind it is by the numbers the told stage keeps. The two
        // stages hold the same dispatches behind each.
        let alike = |pool: &[Candidate; 4], step: &str| {
            let main = scores(&untold, pool);
            assert_eq!(scores(&told, pool), main, "{step}");
            assert_eq!(scores(&told, pool), main, "{step}: by the number");
            for row in pool {
                let (ours, theirs) = (
                    told.held_at(row.id, row.dp_rank),
                    untold.held_at(row.id, row.dp_rank),
                );
                assert_eq!(
                    ours, theirs,
                    "{step}: the dispatches of row {}",
                    row.position
                );
            }
        };
        alike(&pool, "nothing told yet");
        for (row, slot) in pool.iter().zip(slots) {
            tell(&told, slot, row);
        }
        alike(&pool, "each report told");
        // A report told again is no word: its number stands.
        for (row, slot) in pool.iter().zip(slots) {
            tell(&told, slot, row);
        }
        // The report of one request and 2,000 tokens, at 700 a request.
        assert_eq!(kept_of(&told, slots[0], &pool[0]), Some((2_700.0, 1)));
        assert_eq!(kept_of(&told, slots[2], &pool[2]), Some((4_400.0, 2)));
        assert_eq!(kept_of(&told, slots[3], &pool[3]), None, "no count");

        let lease = Lease {
            issued_micros: 2_000,
            tokens: 1_500,
            ..Lease::default()
        };
        for stage in [&told, &untold] {
            stage.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
            stage.on_dispatched(&lease, &target_of(&pool[2]), Side::Single, 0);
        }
        alike(&pool, "a dispatch credited");
        assert_eq!(kept_of(&told, slots[0], &pool[0]), Some((4_900.0, 2)));

        // A newer report reaches a decision before its stage is told it:
        // the row is costed from its parts, and its credit is released.
        pool[0].load = Some(resident(2, 3_600));
        pool[0].load_sampled_at_micros = 3_000;
        assert_eq!(kept_of(&told, slots[0], &pool[0]), None, "the older number");
        alike(&pool, "a newer report shown");
        tell(&told, slots[0], &pool[0]);
        alike(&pool, "the newer report told");
        assert_eq!(kept_of(&told, slots[0], &pool[0]), Some((5_000.0, 2)));

        // A report older than the one kept is late: the number stands.
        let mut late = pool[0];
        late.load = Some(resident(9, 9_000));
        late.load_sampled_at_micros = 2_500;
        tell(&told, slots[0], &late);
        assert_eq!(kept_of(&told, slots[0], &pool[0]), Some((5_000.0, 2)));

        // Another report under the same instant keeps no number: a row
        // that holds the first names that instant too. The next instant's
        // report is kept.
        pool[1].load = Some(resident(3, 4_500));
        tell(&told, slots[1], &pool[1]);
        alike(&pool, "another report at one instant");
        assert_eq!(
            kept_of(&told, slots[1], &pool[1]),
            None,
            "two of one instant"
        );
        pool[1].load_sampled_at_micros = 3_500;
        tell(&told, slots[1], &pool[1]);
        alike(&pool, "the next instant's report");
        assert_eq!(kept_of(&told, slots[1], &pool[1]), Some((6_600.0, 3)));

        // A report that counts no resident tokens keeps no number.
        pool[1].load = Some(uncounted(3));
        pool[1].load_sampled_at_micros = 4_000;
        tell(&told, slots[1], &pool[1]);
        alike(&pool, "a report with no count");
        assert_eq!(kept_of(&told, slots[1], &pool[1]), None);

        // A rank that leaves takes its number and its credit with it.
        for stage in [&told, &untold] {
            stage.on_dispatched(&lease, &target_of(&pool[2]), Side::Single, 0);
            stage.on_membership(&Membership::Removed(pool[2].id));
        }
        assert_eq!(kept_of(&told, slots[2], &pool[2]), None, "the leaver's");
        alike(&pool, "a rank left");
    }

    /// Between a report's tell and the show of its world a decision still
    /// reads the world that stands. A told stage scores that world's rows
    /// as a stage told nothing does, bit for bit: a tell releases nothing,
    /// so the dispatches the newer report reflects are still credited to
    /// the row that holds the report before it. Behind the show it scores
    /// the newer world alike too, with a dispatch made in the window and
    /// one dated before the newer sample. The two stages hold the same
    /// dispatches at every step: each release is a decision's.
    #[test]
    fn a_standing_world_scores_alike_between_a_tell_and_its_show() {
        // Three rows, the row the dispatches land on between a cheap one
        // and a dear one: its score turns on its cost.
        let mut standing = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        standing[0].load = Some(resident(1, 2_000));
        standing[1].load = Some(resident(1, 500));
        standing[2].load = Some(resident(1, 40_000));
        for row in &mut standing {
            row.load_sampled_at_micros = 1_000;
        }
        let lent = Lent::take(&mut standing);
        let slots = [0, 1, 2].map(|at: usize| lent.slot_of(&standing[at]));
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let scores = |stage: &ResidentWork, pool: &[Candidate; 3]| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 3];
            stage.score(&facts, pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        let dispatch = |issued_micros: u64| {
            let lease = Lease {
                issued_micros,
                tokens: 1_500,
                ..Lease::default()
            };
            for stage in [&told, &untold] {
                stage.on_dispatched(&lease, &target_of(&standing[0]), Side::Single, 0);
            }
        };
        let rank = (standing[0].id, standing[0].dp_rank);
        let held = |stage: &ResidentWork| stage.held_at(rank.0, rank.1);
        for (row, slot) in standing.iter().zip(slots) {
            tell(&told, slot, row);
        }
        dispatch(2_000);
        assert_eq!(scores(&told, &standing), scores(&untold, &standing));

        // The next report reflects that dispatch. Its world is told and
        // not shown yet: the standing row keeps the dispatch's credit.
        let mut next = standing;
        next[0].load = Some(resident(2, 3_600));
        next[0].load_sampled_at_micros = 3_000;
        tell(&told, slots[0], &next[0]);
        assert_eq!(
            scores(&told, &standing),
            scores(&untold, &standing),
            "told and not shown"
        );
        assert_eq!(
            (held(&told), held(&untold)),
            (1, 1),
            "a tell releases nothing"
        );

        // A dispatch in the window is credited behind both reports. One a
        // stepped clock dates before the newer sample is credited behind
        // the standing report alone.
        dispatch(3_500);
        dispatch(2_500);
        assert_eq!(
            scores(&told, &standing),
            scores(&untold, &standing),
            "dispatches in the window"
        );
        let number = kept_of(&told, slots[0], &next[0]);
        assert_eq!(number, None, "no number ahead of the show");

        // The world is shown. Its first decision releases what the report
        // reflects, on both stages, and the told one keeps the number from
        // it: the newer report's two requests and 3,600 tokens, and the
        // one dispatch made after its sample.
        assert_eq!(scores(&told, &next), scores(&untold, &next), "shown");
        assert_eq!((held(&told), held(&untold)), (1, 1), "a decision releases");
        assert_eq!(kept_of(&told, slots[0], &next[0]), Some((7_200.0, 3)));
        assert_eq!(
            scores(&told, &next),
            scores(&untold, &next),
            "by the number"
        );
    }

    /// A late report is costed against what the decisions before it
    /// released. A report, a dispatch, a newer report shown and decided,
    /// then a report sampled before that dispatch shown: a stage told
    /// nothing released the dispatch at the newer report's first decision
    /// and counts it no more, and the told stage released it there too. A
    /// dispatch the shown report reflects that lands behind its first
    /// decision is released by the next one: the number steps aside for
    /// that decision, so a late row costed in between still counts it.
    #[test]
    fn a_late_report_scores_alike_behind_a_newer_one_that_was_decided() {
        // Three rows, the row the dispatches land on between a cheap one
        // and a dear one: its score turns on its cost.
        let mut first = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        first[0].load = Some(resident(1, 2_000));
        first[1].load = Some(resident(1, 500));
        first[2].load = Some(resident(1, 40_000));
        for row in &mut first {
            row.load_sampled_at_micros = 1_000;
        }
        let lent = Lent::take(&mut first);
        let slots = [0, 1, 2].map(|at: usize| lent.slot_of(&first[at]));
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let scores = |stage: &ResidentWork, pool: &[Candidate; 3]| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 3];
            stage.score(&facts, pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        let dispatch = |issued_micros: u64| {
            let lease = Lease {
                issued_micros,
                tokens: 1_500,
                ..Lease::default()
            };
            for stage in [&told, &untold] {
                stage.on_dispatched(&lease, &target_of(&first[0]), Side::Single, 0);
            }
        };
        let rank = (first[0].id, first[0].dp_rank);
        let held = |stage: &ResidentWork| stage.held_at(rank.0, rank.1);
        let alike = |pool: &[Candidate; 3], step: &str| {
            assert_eq!(scores(&told, pool), scores(&untold, pool), "{step}");
            assert_eq!(held(&told), held(&untold), "{step}: the dispatches held");
        };
        for (row, slot) in first.iter().zip(slots) {
            tell(&told, slot, row);
        }
        alike(&first, "a report");
        dispatch(2_000);
        alike(&first, "a dispatch");

        // A newer report, told, shown and decided: both stages release
        // the dispatch it reflects.
        let mut newer = first;
        newer[0].load = Some(resident(2, 3_600));
        newer[0].load_sampled_at_micros = 3_000;
        tell(&told, slots[0], &newer[0]);
        alike(&newer, "a newer report decided");
        assert_eq!(held(&untold), 0, "the dispatch is released");
        alike(&newer, "by the number");

        // A report sampled before that dispatch, shown late: the tell
        // drops it, and the row is costed from its parts.
        let mut late = first;
        late[0].load = Some(resident(1, 2_400));
        late[0].load_sampled_at_micros = 1_500;
        tell(&told, slots[0], &late[0]);
        alike(&late, "a late report");

        // The newer report again, and a dispatch it reflects that lands
        // behind its decision. The late row still counts that dispatch,
        // and the newer report's next decision releases it.
        alike(&newer, "the newer report again");
        let number = kept_of(&told, slots[0], &newer[0]);
        assert_eq!(number, Some((5_000.0, 2)), "its number stands");
        dispatch(2_800);
        let number = kept_of(&told, slots[0], &newer[0]);
        assert_eq!(number, None, "the number steps aside");
        assert_eq!(held(&untold), 1, "held until a decision under the report");
        alike(&late, "a late report behind that dispatch");
        alike(&newer, "the next decision under the report");
        assert_eq!(held(&untold), 0, "released");
        let number = kept_of(&told, slots[0], &newer[0]);
        assert_eq!(number, Some((5_000.0, 2)), "the number is kept again");

        // A dispatch dated at the report's own instant is one the report
        // reflects too.
        dispatch(3_000);
        let number = kept_of(&told, slots[0], &newer[0]);
        assert_eq!(number, None, "at the sample's instant");
        alike(&newer, "a dispatch at the sample's instant");
        assert_eq!(held(&untold), 0, "released");

        // A dispatch made after the sample is in the number at once.
        dispatch(3_500);
        assert_eq!(kept_of(&told, slots[0], &newer[0]), Some((7_200.0, 3)));
        alike(&newer, "a dispatch behind the report");
    }

    /// The first decision keeps a number only where the row holds no
    /// dispatch its report reflects, and only of the report the entry
    /// holds. A dispatch dated before the sample that lands between a
    /// decision's release and its word to the stage waits for the next
    /// decision's release: the cell awaits on. A row that holds another
    /// report than the one told is kept no number of.
    #[test]
    fn a_first_decision_keeps_no_number_over_a_dispatch_its_report_reflects() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 3_000;
        let lent = Lent::take(&mut pool);
        let (slot, row) = (lent.slot_of(&pool[0]), pool[0]);
        tell(&stage, slot, &row);
        let lease = Lease {
            issued_micros: 2_800,
            tokens: 1_500,
            ..Lease::default()
        };
        stage.on_dispatched(&lease, &target_of(&row), Side::Single, 0);
        let cell = stage.kept.cell(slot).expect("the slot's cell");
        assert!(cell.awaits(&row), "the report's first decision");

        // The decision's word with that dispatch still held, as where it
        // landed behind the decision's release.
        let since = stage
            .credits
            .entry((row.id, row.dp_rank))
            .expect("the entry");
        stage.settle(Some(cell), &since, &row, slot);
        assert_eq!(
            kept_of(&stage, slot, &row),
            None,
            "a reflected dispatch held"
        );
        assert!(cell.awaits(&row), "the next decision's");

        // A row with another report under the instant is no word of the
        // told one.
        let mut other = row;
        other.load = Some(resident(2, 9_000));
        decide(&stage, &lent, &row);
        assert_eq!(kept_of(&stage, slot, &row), Some((2_700.0, 1)));
        tell(
            &stage,
            slot,
            &Candidate {
                load_sampled_at_micros: 4_000,
                ..row
            },
        );
        other.load_sampled_at_micros = 4_000;
        stage.settle(Some(cell), &since, &other, slot);
        assert_eq!(kept_of(&stage, slot, &other), None, "another report");
    }

    /// A decision that costs a rank from its parts under a later sample
    /// than the kept report's releases dispatches the kept number counts,
    /// as it does on a stage told nothing. The number steps aside, and the
    /// kept report's next decision reads the row as it stands. A world a
    /// decision still holds shows such a sample where a member left and
    /// came back under a new slot, told a report sampled earlier.
    #[test]
    fn a_later_sample_costed_from_its_parts_sends_the_kept_number_back() {
        // The member under test at two places: where the held world shows
        // it, and where it stands once it came back. Its cost lies between
        // a cheap row's and a dear row's, so its score turns on it.
        let mut places = [
            candidate(0, 1),
            candidate(1, 1),
            candidate(2, 1),
            candidate(0, 1),
        ];
        places[3].position = 3;
        places[0].load = Some(resident(1, 2_000));
        places[1].load = Some(resident(1, 500));
        places[2].load = Some(resident(1, 40_000));
        places[3].load = Some(resident(1, 1_000));
        for row in &mut places[..3] {
            row.load_sampled_at_micros = 100_000;
        }
        places[3].load_sampled_at_micros = 20_000;
        let lent = Lent::take(&mut places);
        let slots = [0, 1, 2, 3].map(|at: usize| lent.slot_of(&places[at]));
        let held_world = [places[0], places[1], places[2]];
        let back = [places[3], places[1], places[2]];
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let scores = |stage: &ResidentWork, pool: &[Candidate; 3]| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 3];
            stage.score(&facts, pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        let member = (places[0].id, places[0].dp_rank);
        let held = |stage: &ResidentWork| stage.held_at(member.0, member.1);
        let alike = |pool: &[Candidate; 3], step: &str| {
            let main = scores(&untold, pool);
            assert_eq!(scores(&told, pool), main, "{step}");
            assert_eq!(scores(&told, pool), main, "{step}: by the number");
            assert_eq!(held(&told), held(&untold), "{step}: the dispatches held");
        };
        for (row, slot) in held_world.iter().zip(slots) {
            tell(&told, slot, row);
        }
        alike(&held_world, "the world a decision holds");

        // The member leaves and comes back under another slot, told a
        // report sampled earlier. A dispatch lands behind that report.
        for stage in [&told, &untold] {
            stage.on_membership(&Membership::Removed(member.0));
        }
        tell(&told, slots[3], &back[0]);
        alike(&back, "the member came back");
        let lease = Lease {
            issued_micros: 60_000,
            tokens: 3_000,
            ..Lease::default()
        };
        for stage in [&told, &untold] {
            stage.on_dispatched(&lease, &target_of(&back[0]), Side::Single, 0);
        }
        alike(&back, "a dispatch behind its report");
        assert_eq!(kept_of(&told, slots[3], &back[0]), Some((5_400.0, 2)));

        // The held world's row of the member is sampled later than that
        // dispatch: costed from its parts, it releases it on both stages.
        alike(&held_world, "the world a decision still holds");
        assert_eq!(held(&untold), 0, "the later sample released the dispatch");
        let number = kept_of(&told, slots[3], &back[0]);
        assert_eq!(number, None, "the number steps aside");
        alike(&back, "the member's row as it stands");
        assert_eq!(kept_of(&told, slots[3], &back[0]), Some((1_700.0, 1)));
    }

    /// One stage may be handed both forms of a view: rows beside a lent
    /// table, and rows a host wrote with their own counts. A written row
    /// costed under a later sample than the kept report's releases
    /// dispatches the kept number counts, as it does on a stage told
    /// nothing. The number steps aside there too, and the kept report's
    /// next decision reads the row as it stands.
    #[test]
    fn a_later_sample_in_a_written_view_sends_the_kept_number_back() {
        // The member under test costs between a cheap row and a dear row,
        // so its score turns on its cost.
        let mut rows = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        rows[0].load = Some(resident(1, 1_000));
        rows[1].load = Some(resident(1, 500));
        rows[2].load = Some(resident(1, 40_000));
        for row in &mut rows {
            row.load_sampled_at_micros = 20_000;
        }
        let lent = Lent::take(&mut rows);
        let slots = [0, 1, 2].map(|at: usize| lent.slot_of(&rows[at]));
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let scores = |stage: &ResidentWork, pool: &[Candidate; 3]| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 3];
            stage.score(&facts, pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        // The same rows with no table lent: each row's own count.
        let written = |stage: &ResidentWork, pool: &[Candidate; 3]| {
            let mut out = [0.0f32; 3];
            stage.score(&RequestFacts::default(), pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        let member = (rows[0].id, rows[0].dp_rank);
        let held = |stage: &ResidentWork| stage.held_at(member.0, member.1);
        let alike = |pool: &[Candidate; 3], step: &str| {
            let main = scores(&untold, pool);
            assert_eq!(scores(&told, pool), main, "{step}");
            assert_eq!(scores(&told, pool), main, "{step}: by the number");
            assert_eq!(held(&told), held(&untold), "{step}: the dispatches held");
        };
        for (row, slot) in rows.iter().zip(slots) {
            tell(&told, slot, row);
        }
        alike(&rows, "the world told");
        let lease = Lease {
            issued_micros: 60_000,
            tokens: 3_000,
            ..Lease::default()
        };
        for stage in [&told, &untold] {
            stage.on_dispatched(&lease, &target_of(&rows[0]), Side::Single, 0);
        }
        alike(&rows, "a dispatch behind its report");
        assert_eq!(kept_of(&told, slots[0], &rows[0]), Some((5_400.0, 2)));

        // A host's own rows, the member's sampled later than that
        // dispatch: costed from its parts, the row releases it on both
        // stages.
        let mut later = rows;
        later[0].load_sampled_at_micros = 100_000;
        assert_eq!(
            written(&told, &later),
            written(&untold, &later),
            "a written view"
        );
        assert_eq!(held(&untold), 0, "the later sample released the dispatch");
        assert_eq!(held(&told), 0, "on the told stage too");
        let number = kept_of(&told, slots[0], &rows[0]);
        assert_eq!(number, None, "the number steps aside");
        alike(&rows, "the member's row as it stands");
        assert_eq!(kept_of(&told, slots[0], &rows[0]), Some((1_700.0, 1)));
    }

    /// A stage under a report-age limit is told its reports through the
    /// limit's wrapper and keeps their numbers. A row past the limit is
    /// read as unreported before its cell: it scores as under a stage told
    /// nothing, though the cell still holds its number.
    #[test]
    fn a_row_past_the_report_age_limit_is_read_without_its_kept_number() {
        let mut rows = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        rows[0].load = Some(resident(1, 1_000));
        rows[1].load = Some(resident(1, 500));
        rows[2].load = Some(resident(1, 40_000));
        // The member under test was sampled 8 ms ahead of its peers.
        rows[0].load_sampled_at_micros = 12_000;
        rows[1].load_sampled_at_micros = 20_000;
        rows[2].load_sampled_at_micros = 20_000;
        let lent = Lent::take(&mut rows);
        let slots = [0, 1, 2].map(|at: usize| lent.slot_of(&rows[at]));
        let limited = || WithLoadReportAgeLimit {
            stage: ResidentWork::new(700),
            max_age_ms: 10,
        };
        let (told, untold) = (limited(), limited());
        assert!(told.keeps_loads(), "the wrapper says what its stage keeps");
        let scores = |stage: &WithLoadReportAgeLimit<ResidentWork>, now_micros: u64| {
            let facts = lent.beside(&RequestFacts {
                now_micros,
                ..RequestFacts::default()
            });
            let mut out = [0.0f32; 3];
            stage.score(&facts, &rows, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        for (row, slot) in rows.iter().zip(slots) {
            told.on_signal(&Signal::Load {
                worker: row.id,
                dp_rank: row.dp_rank,
                row: row.load.expect("a report"),
                slot,
                sampled_at_micros: row.load_sampled_at_micros,
                signals: ReportSignals::NONE,
            });
        }

        // Every report within the limit: the first decision keeps the
        // numbers, the second reads them.
        let within = scores(&untold, 21_000);
        assert_eq!(scores(&told, 21_000), within, "within the limit");
        assert_eq!(
            scores(&told, 21_000),
            within,
            "within the limit: by the number"
        );
        let number = kept_of(&told.stage, slots[0], &rows[0]);
        assert_eq!(number, Some((1_700.0, 1)), "told through the wrapper");

        // Two milliseconds on the member's report is past the limit: its
        // row is costed against its peers, as a row with no report is.
        let past = scores(&untold, 23_000);
        assert_ne!(past, within, "the row past the limit scores apart");
        assert_eq!(scores(&told, 23_000), past, "past the limit");
        assert_eq!(
            kept_of(&told.stage, slots[0], &rows[0]),
            number,
            "the cell holds it still"
        );
        assert_eq!(scores(&told, 21_000), within, "within the limit again");
    }

    /// A told stage beside a stage told nothing through a long walk of a
    /// host's events: reports newer, late and under the shown instant,
    /// with a count and without; dispatches dated ahead of the shown
    /// sample and behind it, on the rank the world shows and on the rank
    /// it was before a move; a burst past the credit's bound; requests
    /// that end; leaves, resizes and moved ranks. The host tells each
    /// world before it shows it. Every decision scores alike on both, bit
    /// for bit: over the world that stood, between the tell and the show,
    /// over the world shown, read twice so that the second read is by the
    /// numbers kept, and over a world of some steps back that a decision
    /// still holds. And both hold the same dispatches of every rank at
    /// every step: each release is a decision's.
    #[test]
    fn a_told_stage_walks_beside_a_stage_told_nothing() {
        const ROWS: usize = 5;
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        let mut state = 0x243F_6A88_85A3_08D3u64;
        let mut next = move |bound: u64| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state % bound
        };
        let mut world: Vec<Candidate> = (0..ROWS as u32).map(|at| candidate(at, 0)).collect();
        let mut counts = [0u32; ROWS];
        let slots: Vec<WorkerSlot> = {
            let mut rows = world.clone();
            let lent = Lent::take(&mut rows);
            rows.iter().map(|row| lent.slot_of(row)).collect()
        };
        // A stage's scores of `rows` under `counts`, as bits, and how many
        // of the rows it hands a number it keeps.
        let scores = |stage: &ResidentWork, rows: &[Candidate], counts: &[u32; ROWS]| {
            let mut rows = rows.to_vec();
            for (row, &count) in rows.iter_mut().zip(counts) {
                row.in_flight = count;
            }
            let lent = Lent::take(&mut rows);
            let by_number = rows
                .iter()
                .zip(counts)
                .filter(|&(row, &count)| {
                    let number = kept_of(stage, lent.slot_of(row), row);
                    number.is_some_and(|(_, accounted)| count <= accounted)
                })
                .count();
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; ROWS];
            stage.score(&facts, &rows, &kept(ROWS as u32), &mut out);
            (out.map(f32::to_bits), by_number)
        };
        let held_alike = |world: &[Candidate], step: u32| {
            for row in world {
                for rank in [row.dp_rank, row.dp_rank ^ 1] {
                    let (ours, theirs) = (told.held_at(row.id, rank), untold.held_at(row.id, rank));
                    assert_eq!(
                        ours, theirs,
                        "step {step}: row {} rank {rank}",
                        row.position
                    );
                }
            }
        };
        let (mut by_number, mut reflected, mut windows) = (0usize, 0u32, 0u32);
        let (mut late, mut retired) = (0u32, false);
        // A world some decision still holds: the shown world of up to
        // nine steps back.
        let mut held_world = world.clone();
        let mut now = 1_000_000u64;
        for step in 0..6_000u32 {
            now += 1 + next(2_000);
            let at = next(ROWS as u64) as usize;
            let stood = world.clone();
            let mut made = false;
            let event = if step == 3_000 { 40 } else { next(100) };
            match event {
                // A report: a newer one, another under the shown instant,
                // or a late one; one in ten counts no resident tokens.
                0..=39 => {
                    let running = next(6) as u32;
                    let report = match next(10) {
                        0 => uncounted(running),
                        _ => resident(running, next(8_000) as u32),
                    };
                    let shown = world[at].load_sampled_at_micros;
                    world[at].load_sampled_at_micros = match next(10) {
                        0 => shown.max(1),
                        1 => {
                            late += 1;
                            shown.saturating_sub(1 + next(500)).max(1)
                        }
                        _ => now,
                    };
                    world[at].load = Some(report);
                    counts[at] = running + next(3) as u32;
                    made = true;
                }
                // A dispatch, or at one step a burst past the bound with
                // no decision between: on the row's rank, or on the rank
                // it was before a move. A lease is dated up to three
                // milliseconds back, so some land behind a report that
                // reflects them.
                40..=74 => {
                    let burst = step == 3_000;
                    let mut row = world[at];
                    let was = !burst && next(10) == 0;
                    if was {
                        row.dp_rank ^= 1;
                    }
                    let times = if burst { RANK_CREDIT_KEPT + 3 } else { 1 };
                    for nth in 0..times {
                        let lease = Lease {
                            issued_micros: now + nth - next(3_000),
                            tokens: (next(4) * (1 + next(3_000))) as u32,
                            ..Lease::default()
                        };
                        reflected += u32::from(lease.issued_micros <= row.load_sampled_at_micros);
                        for stage in [&told, &untold] {
                            stage.on_dispatched(&lease, &target_of(&row), Side::Single, 0);
                        }
                        counts[at] += u32::from(!was);
                    }
                    if burst {
                        let held = told.held_at(row.id, row.dp_rank) as u64;
                        retired = held < times;
                    }
                }
                // A request of the row ends.
                75..=84 => counts[at] = counts[at].saturating_sub(1),
                // The member leaves or is resized: its entries go.
                85..=92 => {
                    let id = world[at].id;
                    let change = match next(2) {
                        0 => Membership::Removed(id),
                        _ => Membership::Resized(id),
                    };
                    for stage in [&told, &untold] {
                        stage.on_membership(&change);
                    }
                    made = true;
                }
                // A registration moves the member's rank under its slot.
                _ => {
                    world[at].dp_rank ^= 1;
                    made = true;
                }
            }
            held_alike(&world, step);
            if made {
                // The host tells the world it made, each row that holds a
                // report, before it shows it. Until the show a decision
                // reads the world that stood.
                for (row, &slot) in world.iter().zip(&slots) {
                    if row.load.is_some() {
                        tell(&told, slot, row);
                    }
                }
                let (ours, _) = scores(&told, &stood, &counts);
                let (mains, _) = scores(&untold, &stood, &counts);
                assert_eq!(
                    ours, mains,
                    "step {step}, event {event}: the world that stood"
                );
                held_alike(&world, step);
                windows += 1;
            }
            // The world is shown: its first read, then one by the numbers.
            let (mains, none) = scores(&untold, &world, &counts);
            assert_eq!(none, 0, "a stage told nothing keeps no number");
            let (ours, _) = scores(&told, &world, &counts);
            assert_eq!(ours, mains, "step {step}, event {event} at row {at}: shown");
            let (again, numbers) = scores(&told, &world, &counts);
            assert_eq!(
                again, mains,
                "step {step}, event {event} at row {at}: by the number"
            );
            by_number += numbers;
            held_alike(&world, step);
            // A decision that still holds an older world reads it now.
            let (ours, _) = scores(&told, &held_world, &counts);
            let (mains, _) = scores(&untold, &held_world, &counts);
            assert_eq!(
                ours, mains,
                "step {step}, event {event}: a world still held"
            );
            held_alike(&world, step);
            if step % 9 == 0 {
                held_world.clone_from(&world);
            }
        }
        // The walk took each path often: rows handed a number, dispatches
        // a shown report reflects, late reports, worlds told and not shown
        // yet, and the retirement past the bound.
        assert!(
            by_number > 4_000 && reflected > 100 && late > 100 && windows > 2_000 && retired,
            "{by_number} by a number, {reflected} reflected, {late} late, {windows} windows, retired {retired}"
        );
    }

    /// A report that counts no resident tokens still releases the
    /// dispatches it reflects, beside a lent table as beside none: its
    /// rank would hold its credit to the bound otherwise.
    #[test]
    fn a_report_with_no_count_releases_what_it_reflects_beside_a_lent_table() {
        let mut pool = [candidate(0, 1)];
        pool[0].load = Some(uncounted(1));
        pool[0].load_sampled_at_micros = 3_000;
        let lent = Lent::take(&mut pool);
        let stage = ResidentWork::new(700);
        let lease = Lease {
            issued_micros: 2_000,
            tokens: 1_500,
            ..Lease::default()
        };
        stage.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
        assert_eq!(stage.held_at(pool[0].id, pool[0].dp_rank), 1);
        let facts = lent.beside(&RequestFacts::default());
        let mut out = [0.0f32; 1];
        stage.score(&facts, &pool, &kept(1), &mut out);
        assert_eq!(
            stage.held_at(pool[0].id, pool[0].dp_rank),
            0,
            "the report reflects the dispatch"
        );
    }

    /// Past the credit's bound the row's oldest dispatches retire, and the
    /// number gives up what a retirement takes: it is made of the row's
    /// credit as a stage told nothing reads it, at the first decision and
    /// behind each dispatch.
    #[test]
    fn a_rank_past_the_credits_bound_keeps_the_number_an_untold_stage_reads() {
        // Three rows, the rank past the bound the dearest: the middle
        // row's score turns on that rank's cost.
        let mut pool = [candidate(0, 1), candidate(1, 1), candidate(2, 1)];
        for (row, tokens) in pool.iter_mut().zip([2_000, 500, 40_000]) {
            row.load = Some(resident(1, tokens));
            row.load_sampled_at_micros = 1_000;
        }
        let lent = Lent::take(&mut pool);
        let slot = lent.slot_of(&pool[0]);
        let (told, untold) = (ResidentWork::new(700), ResidentWork::new(700));
        for row in &pool {
            shown(&told, &lent, lent.slot_of(row), row);
        }
        let over = RANK_CREDIT_KEPT + 3;
        for at in 0..over {
            let lease = Lease {
                issued_micros: 2_000 + at,
                tokens: 10,
                ..Lease::default()
            };
            for stage in [&told, &untold] {
                stage.on_dispatched(&lease, &target_of(&pool[0]), Side::Single, 0);
            }
        }
        let held = told.held_at(pool[0].id, pool[0].dp_rank);
        assert!((held as u64) < over, "a retirement ran: {held} held");
        assert_eq!(held, untold.held_at(pool[0].id, pool[0].dp_rank));
        let scores = |stage: &ResidentWork| {
            let facts = lent.beside(&RequestFacts::default());
            let mut out = [0.0f32; 3];
            stage.score(&facts, &pool, &kept(3), &mut out);
            out.map(f32::to_bits)
        };
        assert!(kept_of(&told, slot, &pool[0]).is_some(), "by the number");
        let scored = scores(&told);
        assert_eq!(scored, scores(&untold));
        assert!(
            scored[2] != scored[0] && scored[2] != scored[1],
            "the middle row stands between: {scored:?}"
        );
        let (_, accounted) = kept_of(&told, slot, &pool[0]).expect("a number");
        assert_eq!(u64::from(accounted), 1 + held as u64);
    }

    /// A rank whose slot another rank took is bound to no cell, and a
    /// dispatch to it writes no number. Told that report again under a
    /// slot, it keeps the number of the report and of every dispatch the
    /// row holds: the number is made of the row's credit at the report's
    /// first decision.
    #[test]
    fn a_rank_bound_again_keeps_the_dispatches_it_took_unbound() {
        let stage = ResidentWork::new(700);
        let mut pool = [candidate(0, 0), candidate(1, 0)];
        pool[0].load = Some(resident(1, 2_000));
        pool[0].load_sampled_at_micros = 1_000;
        pool[1].load = Some(resident(3, 6_000));
        pool[1].load_sampled_at_micros = 1_500;
        let lent = Lent::take(&mut pool);
        let (first, second) = (lent.slot_of(&pool[0]), lent.slot_of(&pool[1]));
        let (rank, other) = (pool[0], pool[1]);
        shown(&stage, &lent, first, &rank);
        assert_eq!(kept_of(&stage, first, &rank), Some((2_700.0, 1)));
        // Another rank takes the slot: the first is bound to none.
        tell(&stage, first, &other);
        assert_eq!(kept_of(&stage, first, &rank), None, "the slot is another's");
        let lease = Lease {
            issued_micros: 3_000,
            tokens: 1_500,
            ..Lease::default()
        };
        stage.on_dispatched(&lease, &target_of(&rank), Side::Single, 0);
        // Told its report again, under a second slot, and shown at that
        // slot's place.
        let there = Candidate {
            position: other.position,
            ..rank
        };
        shown(&stage, &lent, second, &there);
        assert_eq!(kept_of(&stage, second, &rank), Some((4_900.0, 2)));
    }
}
