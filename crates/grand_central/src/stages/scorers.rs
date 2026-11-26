//! The scorers written for no one policy. Every stateful one keeps its state
//! behind the lifecycle alone, and none allocates per decision.
//!
//! The credit of the sends since a worker's last report (`RankCredit`,
//! `Credits`) is in this file too. `ResidentWork` scores with it, and so does
//! the scorer of `least_load`, through the sent record in `tracking`.

use std::{
    cell::Cell,
    collections::VecDeque,
    hash::BuildHasher,
    sync::{
        atomic::{AtomicU16, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
};

use arc_swap::ArcSwap;
use parking_lot::{Mutex, MutexGuard};
use rustc_hash::FxBuildHasher;

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

    fn balances_load_against_affinity(&self) -> bool {
        self.stage.balances_load_against_affinity()
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
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
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
        for (slot, &i) in out.iter_mut().zip(kept) {
            *slot = facts.in_flight(&candidates[i as usize]) as f32;
        }
        normalize_costs(out);
        TieBand::NORMALIZED
    }
}

/// The owner of the request's session key scores `1.0`, every other `0.0`.
#[derive(Debug)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
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

/// Fewest session keys with a request in flight on the worker (the host's
/// count): the `min_group` assignment of a sticky session.
#[derive(Debug, Default)]
#[expect(clippy::exhaustive_structs, reason = "no field: its name is its value")]
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
            // A case counts the dispatches that land ahead of a lane's end.
            #[cfg(test)]
            tests::LANDED_AHEAD.set(tests::LANDED_AHEAD.get() + 1);
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

    /// The dispatches the row holds, each with its instant and its size,
    /// lane by lane. Read under one hold of every lane's lock, as a
    /// release holds them: none lands and none goes meanwhile.
    fn held_sends(&self) -> Vec<(u64, SendSize)> {
        // A case lets a row's dispatches go here, ahead of the read.
        #[cfg(test)]
        if let Some(ahead) = tests::AHEAD_OF_A_COPYS_READ.get() {
            ahead(self);
        }
        let held_lanes = self.lock_lanes();
        let lanes = held_lanes.iter();
        lanes.flat_map(|lane| lane.0.iter().copied()).collect()
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

/// A candidate row's identity in the tally: the worker and the DP rank. A row
/// is one worker or one rank of a data-parallel worker, and a rank's report
/// and credit are its own (design 4.1: a rank is a candidate), so two ranks of
/// one worker never share an entry.
type RankKey = (WorkerId, u16);

/// A slot of a [`RankTable`]: a rank beside its entry, or free.
type RankSlot = Option<(RankKey, Arc<RankCredit>)>;

/// A tally's entries by rank, as a decision's row loop reads them. Each slot
/// holds a rank beside its entry: a read starts at the slot the pair's hash
/// names and walks to the rank or to a free slot, where a general map's
/// probe of a group of slots cost a least-load decision about 3 % at 72
/// rows. At most half of the slots are held, so a walk ends.
#[derive(Clone, Debug, Default)]
pub(crate) struct RankTable {
    /// A power of two of slots, or none.
    slots: Box<[RankSlot]>,
    held: usize,
}

impl RankTable {
    /// The slot a walk for `rank` starts at, before the table's mask: the
    /// hash a map takes of the pair, so ids a program states in a pattern
    /// start apart.
    #[inline(always)]
    fn start(rank: &RankKey) -> usize {
        FxBuildHasher.hash_one(rank) as usize
    }

    #[inline(always)]
    pub(crate) fn get(&self, rank: &RankKey) -> Option<&Arc<RankCredit>> {
        // A count nothing was sent under answers before it hashes.
        if self.slots.is_empty() {
            return None;
        }
        let mask = self.slots.len() - 1;
        let mut at = Self::start(rank) & mask;
        loop {
            match self.slots.get(at)? {
                Some((held, since)) if held == rank => return Some(since),
                Some(_) => at = (at + 1) & mask,
                None => return None,
            }
        }
    }

    pub(crate) fn contains_key(&self, rank: &RankKey) -> bool {
        self.get(rank).is_some()
    }

    #[inline(always)]
    pub(crate) fn is_empty(&self) -> bool {
        self.held == 0
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &(RankKey, Arc<RankCredit>)> {
        self.slots.iter().flatten()
    }

    pub(crate) fn keys(&self) -> impl Iterator<Item = &RankKey> {
        self.iter().map(|(rank, _)| rank)
    }

    /// Enters `since` under `rank`, which the table does not hold.
    fn insert(&mut self, rank: RankKey, since: Arc<RankCredit>) {
        if (self.held + 1) * 2 > self.slots.len() {
            let entries = std::mem::take(&mut self.slots);
            self.slots = vec![None; (entries.len() * 2).max(4)].into();
            self.held = 0;
            for (rank, since) in entries.into_vec().into_iter().flatten() {
                self.insert(rank, since);
            }
        }
        let mask = self.slots.len() - 1;
        let mut at = Self::start(&rank) & mask;
        while self.slots[at].is_some() {
            at = (at + 1) & mask;
        }
        self.slots[at] = Some((rank, since));
        self.held += 1;
    }

    /// The table of the entries `stays` keeps, laid anew in as many slots,
    /// and in none where no entry stays: a walk stops at a free slot, so
    /// none is freed in place.
    fn retained(&self, mut stays: impl FnMut(&RankKey, &Arc<RankCredit>) -> bool) -> Self {
        let mut next = Self {
            slots: vec![None; self.slots.len()].into(),
            held: 0,
        };
        for (rank, since) in self.iter() {
            if stays(rank, since) {
                next.insert(*rank, Arc::clone(since));
            }
        }
        if next.held == 0 {
            next.slots = Box::default();
        }
        next
    }
}

/// The since-report dispatch credit of a load scorer: per rank row, the
/// dispatches no report reflects yet. A decision and a dispatch to a rank the
/// tally knows load the table without a shared write (a read-write lock's
/// reader count is one, and eight deciders queued on it); a rank's first
/// dispatch and a membership change replace the table whole, which is rare.
/// Within a row the dispatches take per-thread lanes: deciders that read the
/// same minimum book the same rank, and one lock there queued them too.
#[derive(Debug)]
pub(crate) struct Credits {
    pub(crate) tally: ArcSwap<RankTable>,
}

impl Default for Credits {
    fn default() -> Self {
        Self {
            tally: ArcSwap::from_pointee(RankTable::default()),
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
            let mut next = RankTable::clone(tally);
            next.insert(rank, Arc::new(RankCredit::default()));
            Arc::new(next)
        });
        self.tally.load().get(&rank).map(Arc::clone)
    }

    /// The tally without the ranks `gone` names: the table laid anew where a
    /// rank goes and kept as it stands where none does, as a membership change
    /// is rare and a decision never waits on it.
    fn forget(&self, gone: impl Fn(&RankKey) -> bool) {
        self.tally.rcu(|tally| {
            // A change that takes no rank keeps the table it found.
            if !tally.keys().any(&gone) {
                return Arc::clone(tally);
            }
            Arc::new(tally.retained(|rank, _| !gone(rank)))
        });
    }

    /// Holds from here on a copy of what the `standing` tallies hold: of
    /// each rank, the dispatches of the tally that holds the most there.
    /// A tally takes every dispatch its engine is told and lets go a
    /// prefix by time, so the fullest row of a rank holds what the other
    /// rows of it hold, but for a dispatch an earlier copy missed: a row
    /// that came of a copy lacks it, and so does a copy made of that row.
    /// Every standing row of a rank is read, one row's lanes held at a
    /// time for the read of its dispatches alone, and the longest read is
    /// the one copied. The rows are compared by what was read under their
    /// locks: a release that crosses the copy takes no dispatch between
    /// the choice of a row and its read. The reads of a rank's rows are of
    /// different instants: a dispatch told between two of them is in the
    /// copy only where the longest read holds it. Each dispatch enters its
    /// row here as a send does ([`RankCredit::record`]), at its instant
    /// and its size: a row past its bound lets its oldest go. They enter
    /// oldest first, so each lands at its lane's end. What this tally held
    /// until then is let go, and it shares no row with the standing ones.
    pub(crate) fn copy_fullest<'a>(&self, standing: impl Iterator<Item = &'a Self>) {
        let tallies: Vec<_> = standing.map(|credits| credits.tally.load_full()).collect();
        let mut next = RankTable::default();
        for (at, tally) in tallies.iter().enumerate() {
            for (rank, since) in tally.iter() {
                // A rank is entered where the first tally that holds it is
                // read, from the fullest row any tally holds of it.
                if tallies[..at].iter().any(|read| read.contains_key(rank)) {
                    continue;
                }
                let later = tallies[at + 1..].iter();
                let rows = later.filter_map(|tally| tally.get(rank));
                let mut fullest = rows.fold(since.held_sends(), |fullest, row| {
                    let sends = row.held_sends();
                    if sends.len() > fullest.len() {
                        sends
                    } else {
                        fullest
                    }
                });
                fullest.sort_unstable_by_key(|&(at_micros, _)| at_micros);
                let copy = RankCredit::default();
                for (at_micros, size) in fullest {
                    copy.record(at_micros, size);
                }
                next.insert(*rank, Arc::new(copy));
            }
        }
        self.tally.store(Arc::new(next));
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
    /// that worker's next report.
    ///
    /// A rank the tally does not know is entered only where `present`
    /// says its row still stands, so a late dispatch enters no rank
    /// behind a removal. A rank the tally knows takes the dispatch
    /// unasked: a removal forgets the worker's entries, and a join of the
    /// id forgets what entered under it since.
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

    /// Drops the ranks a membership change takes away, and at a join the
    /// ranks held under the joining id.
    pub(crate) fn membership(&self, change: &Membership<'_>) {
        match change {
            // Every rank of the worker goes with it. A resized worker's ranks
            // are a new set: what a vanished rank held would never release,
            // and a kept rank's next report is one interval away. Rare, so a
            // walk over the tally is cheaper than a second index per worker.
            Membership::Removed(id)
            | Membership::Replaced { old: id, .. }
            | Membership::Resized(id) => {
                self.forget(|(worker, _)| worker == id);
            }
            // A join drops what a dispatch told behind the id's removal
            // entered: the worker that joins took none of those sends. A
            // send to the joining worker told ahead of this word goes with
            // them: an id alone does not tell the two apart. A join that
            // finds no rank of its id copies no map.
            Membership::Added(id) => {
                if self.tally.load().keys().any(|(worker, _)| worker == id) {
                    self.forget(|(worker, _)| worker == id);
                }
            }
            // A snapshot (every worker of every pool, after a resync) stands
            // for removals that may never have arrived: only its ids keep a
            // tally. Rare, so a sorted copy is cheaper than a scan per row.
            Membership::Snapshot(ids) => {
                let mut sorted: Vec<u64> = ids.iter().map(|w| w.0).collect();
                sorted.sort_unstable();
                self.forget(|(worker, _)| sorted.binary_search(&worker.0).is_err());
            }
            Membership::Relabelled(_) | Membership::Left { .. } => {}
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
}

impl ResidentWork {
    pub fn new(request_tokens: u32) -> Self {
        Self {
            request_tokens,
            mean_prefill_tokens: DEFAULT_MEAN_PREFILL_TOKENS,
            credits: Credits::default(),
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
        for (slot, &i) in out.iter_mut().zip(kept) {
            if *slot == PENDING {
                let in_flight = facts.in_flight(&candidates[i as usize]) as f32;
                *slot = if fleet_reports {
                    peer_baseline + in_flight * in_flight_work
                } else {
                    in_flight
                };
            }
        }
    }
}

impl ResidentWork {
    /// The cost of `row` by its report and its credit under one load of
    /// the tally, or the mark of a row costed after the reporting peers'
    /// baseline is known. A report releases the dispatches it reflects
    /// whatever the score reads of them: a rank whose report counts no
    /// resident tokens would hold its credit at the bound. The row's
    /// requests in flight are asked for where the row is costed, and of no
    /// row that is marked. Inlined into the row loop of the score: as a
    /// call it read 19 instructions a row more.
    #[inline(always)]
    fn cost_reported(
        &self,
        tally: &RankTable,
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
            // A row that holds no dispatch at or before its sample reads
            // the word alone: the release is a call.
            if since.reaches(row.load_sampled_at_micros) {
                return Self::release(since, row);
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

    /// Releases the dispatches `row`'s report reflects, and hands back the
    /// held token-work and requests behind the release. Cold and out of
    /// line: a decision reads a row far more often than a report releases
    /// from it.
    #[cold]
    #[inline(never)]
    fn release(since: &RankCredit, row: &Candidate) -> HeldCredit {
        RankCredit::credit_of(since.release_through(row.load_sampled_at_micros))
    }
}

impl ResidentWork {
    /// The rows' costs, and how many of them report: every row by its
    /// report and its credit, under one load of the tally. Out of line:
    /// inlined into the score, the bench's cell `resident_work` ran 103
    /// instructions a decision more at 64 workers and 10 fewer at 8.
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
                since.record(at_micros, SendSize::stated(tokens));
            },
        );
    }

    fn on_membership(&self, change: &Membership<'_>) {
        self.credits.membership(change);
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
        // A row's report is read through `load`: a row past a decision's
        // cutoff has none, so no tally is read for it.
        let reporting = self.cost_own(candidates, kept, out, load);
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
        in_flight: u32,
    ) -> RowCost {
        let sum = 0.0 + f64::from(in_flight) * self.terms.in_flight;
        let missing = || {
            self.request_cost.map_or(RowCost::Sum(sum), |price| {
                RowCost::OverLeast(f64::from(in_flight) * price)
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
    ) -> (CostBounds, bool) {
        let mut bounds = CostBounds::EMPTY;
        let mut pending = false;
        costs.extend(kept.iter().map(|&i| {
            let row = &candidates[i as usize];
            let cost = self.cost_of::<RESIDENT, WAITING, KV_OR_RUNNING>(load(row), counts(row));
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
    ) -> (CostBounds, bool) {
        let resident = self.terms.resident_tokens != 0.0;
        let waiting = self.terms.waiting_uncached_tokens != 0.0;
        match (resident, waiting) {
            (false, false) => {
                self.sum_rows::<false, false, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (false, true) => {
                self.sum_rows::<false, true, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (true, false) => {
                self.sum_rows::<true, false, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
            (true, true) => {
                self.sum_rows::<true, true, KV_OR_RUNNING>(candidates, kept, costs, load, counts)
            }
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
        // Every row carries its count, and the walk reads it where it
        // costs the row.
        let counts = |row: &Candidate| facts.in_flight(row);
        if self.terms.names_kv_or_running() {
            self.walk::<true>(candidates, kept, costs, load, counts)
        } else {
            self.walk::<false>(candidates, kept, costs, load, counts)
        }
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
                        *cost = RowCost::OverLeast(f64::from(facts.in_flight(row)) * price);
                        pending = true;
                    }
                    None => bounds.include(priced),
                }
            }
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
#[path = "../../tests/unit/stages/scorers/tests.rs"]
pub(crate) mod tests;
