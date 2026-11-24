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
        Arc, OnceLock,
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
        dispatch_credit::{ComparedReports, DispatchCounts, DispatchCredit},
        in_flight::with_counts,
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
    /// stated token-work ([`Self::counts_released`]). A word of its own,
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
    /// What a count that ends a send only at a changed report keeps of
    /// the row's reports, made at such a count's first compare. No other
    /// count reads or writes it: under one, the entry holds the empty
    /// cell alone.
    pub(crate) reports: OnceLock<Box<ComparedReports>>,
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
            reports: OnceLock::new(),
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
    ///
    /// `release` is the release of the count that reads: it runs where
    /// the row holds a dispatch made at or before `through_micros`. A
    /// count that ends a send only at a changed report compares the row's
    /// report there.
    #[inline(always)]
    pub(crate) fn counts_released(
        &self,
        through_micros: u64,
        release: impl FnOnce(&Self),
    ) -> DispatchCounts {
        if self.reaches(through_micros) {
            release(self);
        }
        let held = self.held.load(Ordering::Acquire);
        DispatchCounts {
            // The word's low bits: the narrower type cuts none of them.
            requests: (held & REQUEST_MASK) as u32,
            tokens: (held >> REQUEST_BITS) + self.estimated(),
        }
    }

    /// [`Self::counts_released`] under the release of a count that sets no
    /// rule on a changed report: a test's read of a row.
    #[cfg(test)]
    pub(crate) fn counts_through(&self, through_micros: u64) -> DispatchCounts {
        self.counts_released(through_micros, |since| {
            since.release_through(through_micros);
        })
    }

    /// The held requests and stated token-work of [`Self::counts_released`]
    /// with the count of the held dispatches of no tokens in the
    /// estimates' place: three loads too. The held word is loaded with
    /// `Acquire`, against the `Release` of a release's and a retirement's
    /// subtraction: the count is read as it stood at that write or later,
    /// so no read finds more dispatches of no tokens than dispatches
    /// across a release, on any target.
    #[inline(always)]
    pub(crate) fn sent_through(&self, through_micros: u64) -> SentCredit {
        self.sent_released(through_micros, |since| {
            since.release_through(through_micros);
        })
    }

    /// [`Self::sent_through`] with the release of the count that reads
    /// ([`Self::counts_released`]).
    #[inline(always)]
    pub(crate) fn sent_released(
        &self,
        through_micros: u64,
        release: impl FnOnce(&Self),
    ) -> SentCredit {
        if self.reaches(through_micros) {
            release(self);
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
    /// [`Self::counts_released`] counts. Lane by
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
    ///
    /// `ends` is asked there, under the hold, for the instant the count
    /// that reads ends the row's dispatches through: `through_micros`
    /// itself, or the one a count that ends a send only at a changed
    /// report makes of the row's reports, and `None` where it ends none.
    pub(crate) fn each_released(
        &self,
        through_micros: u64,
        ends: impl FnOnce(&Self) -> Option<u64>,
        mut each: impl FnMut(u64, u32),
    ) {
        let mut held_lanes = self.lock_lanes();
        if self.reaches(through_micros) {
            // An instant before the row's oldest dispatch takes none, and
            // such a read stores nothing on the row's line either.
            if let Some(ends) = ends(self).filter(|&ends| self.reaches(ends)) {
                self.release_held(&mut held_lanes, ends);
            }
        }
        for &(at_micros, size) in held_lanes.iter().flat_map(|dispatches| &dispatches.0) {
            each(at_micros, size.tokens());
        }
    }

    /// Whether the row holds a dispatch made at or before the sample: one
    /// a report sampled then reflects, which its release takes.
    #[inline(always)]
    pub(crate) fn reaches(&self, sampled_at_micros: u64) -> bool {
        self.oldest_micros.load(Ordering::Acquire) <= sampled_at_micros
    }

    /// [`Self::each_released`] under a count that sets no rule on a
    /// changed report: a test's list of a row.
    #[cfg(test)]
    pub(crate) fn each_through(&self, through_micros: u64, each: impl FnMut(u64, u32)) {
        self.each_released(through_micros, |_| Some(through_micros), each);
    }

    /// Whether a lane's front is a dispatch made at or before the
    /// instant, by the fronts the lanes publish, read with no lock. The
    /// row's oldest instant ([`Self::reaches`]) can stand below every
    /// dispatch the row holds, behind a retirement: a count that keeps
    /// what a read shows it asks here first.
    pub(crate) fn holds_through(&self, through_micros: u64) -> bool {
        self.lanes
            .iter()
            .any(|lane| lane.oldest_micros.load(Ordering::Acquire) <= through_micros)
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
    pub(crate) fn release_through(&self, sampled_at_micros: u64) -> u64 {
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
    /// ranks held under the joining id, each entry that goes shown to
    /// `dropped` first.
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
            // A join drops what a dispatch told behind the id's removal
            // entered: the worker that joins took none of those sends. A
            // send to the joining worker told ahead of this word goes with
            // them: an id alone does not tell the two apart. A join that
            // finds no rank of its id copies no map.
            Membership::Added(id) => {
                if self.tally.load().keys().any(|(worker, _)| worker == id) {
                    self.forget(|(worker, _)| worker == id, dropped);
                }
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
    /// report and its credit, under one load of the tally.
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
        self.credits.membership(change, |_| {});
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
        // Every row carries its count: `written` is true, and the walk
        // reads it.
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

    use super::*;
    use crate::{
        document::spec::{LeastCostParams, WarmupSliceParams},
        engine::{
            decision::{Pick, Targets},
            framework::{Picker, Shortlist},
            view::{Health, LoadRow},
        },
        policies::cache_aware::{Gates, WarmupSlice},
        stages::pickers::{target_of, Best},
        tracking::lease::{EventKind, TransportClass},
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
            fn balances_load_against_affinity(&self) -> bool {
                true
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
        assert!(bare.balances_load_against_affinity());
        assert!(wrapped.balances_load_against_affinity());
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
    fn the_session_scorer_reads_the_rows_own_bit() {
        let mut pool = [candidate(0, 0), candidate(1, 0), candidate(2, 0)];
        let facts = RequestFacts::default();
        let mut scores = [0.0f32; 3];
        SessionAffinity.score(&facts, &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 0.0, 0.0]);
        pool[1].session_hit = true;
        SessionAffinity.score(&facts, &pool, &kept(3), &mut scores);
        assert_eq!(scores, [0.0, 1.0, 0.0]);
        // The scorer is handed the rows kept: a row is read by its index.
        let mut scores = [0.0f32; 2];
        SessionAffinity.score(&facts, &pool, &[1, 2], &mut scores);
        assert_eq!(scores, [1.0, 0.0]);
    }
}
