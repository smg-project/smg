//! The credit of the sends to one rank since its last report, and the table
//! that finds a rank's credit.
//!
//! `RankCredit` holds one rank's sends, `RankTable` finds a rank's entry and
//! `Credits` is the tally built of the two. The sent record beside this file
//! (`dispatch_credit`) keeps one tally for a profile's count of sends, and
//! the `resident-work` scorer keeps a tally of its own.

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
    engine::{decision::Target, view::WorkerId},
    tracking::lease::{Lease, Membership, Side},
};

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
    /// at a release and at a retirement ([`Self::unprice`]), so it never
    /// counts a dispatch the held word does not. The held word's
    /// subtraction there is a `Release` and the read of both words loads
    /// the held word with `Acquire`, so a reader finds that order on every
    /// target. A read loads the two words apart: across a record or a
    /// retirement it may see one without the other for the dispatch in
    /// flight, and across a release for each dispatch that its held word
    /// counts and that goes between its two loads.
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
    /// retirement it may miss the estimate of a dispatch the held word it
    /// loaded still counts, and reads that send at no tokens: at a
    /// release, each such dispatch that goes between its two loads. It
    /// never adds the estimate of a dispatch the held word has let go.
    estimated_held: AtomicU64,
    /// The lowest of the lanes' front instants; `u64::MAX` when none. A
    /// dispatch lowers it under its lane's lock; a release recomputes it
    /// under every lane's lock, so it never reads above any lane's front.
    oldest_micros: AtomicU64,
    /// Each lane's front instant, as the lane published it under its lock.
    fronts: LaneFronts,
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
    pub(crate) const fn tokens(self) -> u32 {
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
/// The instant at its front stands beside the other lanes' ([`LaneFronts`]).
#[derive(Debug, Default)]
#[repr(align(64))]
struct DispatchLane {
    dispatches: Mutex<Dispatches>,
}

/// The instant at the front of each of a row's lanes, `u64::MAX` when none,
/// on one cache line of their own: a retirement reads all of them, and a
/// lane's own line stays with the threads that append to it and with the
/// retirement that takes its front.
#[derive(Debug)]
#[repr(align(64))]
struct LaneFronts([AtomicU64; DISPATCH_LANES]);

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
    pub(crate) const NONE: Self = Self {
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

/// The sends a row holds and their prompt tokens: the two numbers a
/// document's `sent.` terms read
/// ([`DispatchCreditPass::counts`](crate::tracking::DispatchCreditPass::counts)).
/// A send that stated no token count is one of `requests` and adds its
/// estimate, the bytes of its text over four:
/// [`CreditedDispatches`](crate::tracking::CreditedDispatches) states the
/// stated tokens and the estimates apart, and how many such sends there
/// are. The pair is closed: a further number goes on
/// [`CreditedDispatches`](crate::tracking::CreditedDispatches).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[expect(clippy::exhaustive_structs, reason = "the pair is closed: see its doc")]
pub struct DispatchCounts {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens of the sends: the count each one stated, and for
    /// a send that stated none the bytes of its text over four, each held
    /// under 2^31. A send of no tokens and no text adds none. A stage that
    /// reads this number is to ask for each request's size
    /// ([`Needs::prompt_size`](crate::document::Needs::prompt_size)), as
    /// the document's two stages do where a term names `sent.tokens`:
    /// where no stage asks, a host that can send a request on without
    /// reading its prompt may state no bytes of its text.
    pub tokens: u64,
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

impl Default for LaneFronts {
    fn default() -> Self {
        Self(std::array::from_fn(|_| AtomicU64::new(u64::MAX)))
    }
}

impl LaneFronts {
    /// Publishes the front of lane `at`, under that lane's lock. Under the
    /// largest value where the lane holds a dispatch: that value says an
    /// empty lane alone, to a retirement that reads the fronts with no lock.
    fn publish(&self, at: usize, dispatches: &Dispatches) {
        let oldest = dispatches
            .oldest()
            .map_or(u64::MAX, |front| front.min(u64::MAX - 1));
        self.0[at].store(oldest, Ordering::Release);
    }
}

impl Default for RankCredit {
    fn default() -> Self {
        Self {
            held: AtomicU64::new(0),
            unsized_held: AtomicU16::new(0),
            estimated_held: AtomicU64::new(0),
            oldest_micros: AtomicU64::new(u64::MAX),
            fronts: LaneFronts::default(),
            lanes: Default::default(),
        }
    }
}

/// A dispatch on its way out of a row: the held word still counts it, and
/// its price is out of the row's other two words. Where it stated no size,
/// [`RankCredit::unprice`] took its estimate out of the estimates' word
/// and its mark out of the count of no stated size. One that stated its
/// size has no such price, and the release of a row that holds no dispatch
/// of no stated size makes this value with no first step
/// ([`RankCredit::unpriced_stated`]). The held word's step takes the value:
/// the dispatch leaves that word behind its price, or a retirement at the
/// row's bound keeps it and puts its price back.
#[must_use = "the held word still counts the dispatch"]
struct Unpriced(SendSize);

/// A row's held word as one load found it ([`RankCredit::held_word`]). The
/// loads of the row's other two words take this value: a read hands them
/// the one its own load made, so it loads them behind the held word.
struct HeldWord(u64);

impl RankCredit {
    /// One dispatch as the held word counts it.
    fn weight(tokens: u32) -> u64 {
        u64::from(tokens) << REQUEST_BITS | 1
    }

    /// Holds one dispatch of `size` made at `at_micros`. A dispatch that
    /// stated no size is counted as one, and its estimate is kept beside
    /// the stated token-work.
    pub(crate) fn record(&self, at_micros: u64, size: SendSize) {
        let at = lane_of_this_thread();
        let mut dispatches = self.lanes[at].dispatches.lock();
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
            self.fronts.publish(at, &dispatches);
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

    /// The first step of a dispatch's leaving the row, at a release and at
    /// a retirement: where it stated no size, its estimate leaves the
    /// estimates' word and its mark the count of no stated size. The held
    /// word still counts it, and that word's step takes the value handed
    /// back ([`Self::take_out`], [`Self::take_out_past_bound`]).
    #[inline(always)]
    fn unprice(&self, size: SendSize) -> Unpriced {
        if size.is_unsized() {
            self.estimated_held
                .fetch_sub(u64::from(size.estimate()), Ordering::Relaxed);
            self.unsized_held.fetch_sub(1, Ordering::Relaxed);
        }
        Unpriced(size)
    }

    /// The value of a dispatch that stated its size, made with no first
    /// step: such a dispatch has no price to leave. A debug build checks
    /// that the size is stated.
    #[inline(always)]
    fn unpriced_stated(size: SendSize) -> Unpriced {
        debug_assert!(!size.is_unsized(), "the dispatch stated its size");
        Unpriced(size)
    }

    /// The second step at a release: takes the dispatch its lane let go
    /// out of the held word. The subtraction is a `Release`: a read that
    /// loads the word with `Acquire` finds every write made ahead of it,
    /// the count of the dispatches of no tokens among them
    /// ([`Self::sent_through`]).
    #[inline(always)]
    fn take_out(&self, gone: Unpriced) {
        let Unpriced(size) = gone;
        self.held
            .fetch_sub(Self::weight(size.tokens()), Ordering::Release);
    }

    /// The second step at a retirement: takes the dispatch out of the held
    /// word where the row stands past its bound, and says whether it did.
    /// The bound and the subtraction are one step of that word: a
    /// retirement under another lane's lock reads the same word, and one
    /// of the two takes the last dispatch past the bound. Where the row
    /// stands at its bound the dispatch stays, and its price goes back
    /// into its two words.
    #[inline(always)]
    fn take_out_past_bound(&self, gone: Unpriced) -> bool {
        let Unpriced(size) = gone;
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
            return false;
        }
        true
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
            let (oldest, at) =
                self.fronts
                    .0
                    .iter()
                    .enumerate()
                    .fold((u64::MAX, 0), |low, (at, front)| {
                        let front = front.load(Ordering::Acquire);
                        if front < low.0 {
                            (front, at)
                        } else {
                            low
                        }
                    });
            let mut dispatches = self.lanes[at].dispatches.lock();
            // The bound read again under the lock: a release since this
            // dispatch landed may have taken the row to it, and then the
            // newer credits stay.
            if self.held.load(Ordering::Relaxed) & REQUEST_MASK <= RANK_CREDIT_KEPT {
                return;
            }
            // Under the lock the lane's published front is its own. One
            // past the instant read means the one read went to another
            // dispatch's retirement, and the lanes are read again.
            let front = self.fronts.0[at].load(Ordering::Relaxed);
            if front > oldest {
                continue;
            }
            let Some(&(_, size)) = dispatches.0.front() else {
                continue;
            };
            // The estimate and the count of no stated size first and the
            // held word next, as a release lowers them. Where a retirement
            // under another lane's lock took the last dispatch past the
            // bound, this one stays.
            let gone = self.unprice(size);
            if !self.take_out_past_bound(gone) {
                return;
            }
            dispatches.0.pop_front();
            self.fronts.publish(at, &dispatches);
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
        let held = self.held_word();
        self.counts_behind(&held)
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
        let held = self.held_word();
        self.sent_behind(&held)
    }

    /// The four numbers of a whole read once the dispatches made at or
    /// before `through_micros` are released: the three of
    /// [`Self::sent_through`], and the estimates of the held dispatches
    /// that stated no size, summed. Inlined into a reader's row loop: four
    /// loads and a compare a row, the row's oldest instant, the held word
    /// and, behind it, the count and the estimates, with the release
    /// behind a call.
    #[inline(always)]
    pub(crate) fn credited_through(&self, through_micros: u64) -> (SentCredit, u64) {
        if self.reaches(through_micros) {
            self.release_through(through_micros);
        }
        let held = self.held_word();
        (self.sent_behind(&held), self.estimated_behind(&held))
    }

    /// The first step of a read: the held word, loaded with `Acquire`
    /// against the `Release` of a release's and a retirement's
    /// subtraction. A read hands the value to the loads of the row's other
    /// two words, and so finds each as it stood at that subtraction or
    /// later.
    #[inline(always)]
    fn held_word(&self) -> HeldWord {
        HeldWord(self.held.load(Ordering::Acquire))
    }

    /// The second step of [`Self::counts_through`]: the requests the held
    /// word counts and their tokens, stated or estimated, with the
    /// estimates' word loaded behind it.
    #[inline(always)]
    fn counts_behind(&self, held: &HeldWord) -> DispatchCounts {
        DispatchCounts {
            // The word's low bits: the narrower type cuts none of them.
            requests: (held.0 & REQUEST_MASK) as u32,
            tokens: (held.0 >> REQUEST_BITS) + self.estimated_behind(held),
        }
    }

    /// The second step of [`Self::sent_through`]: the requests and the
    /// stated tokens the held word counts, with the count of the
    /// dispatches of no tokens loaded behind it.
    #[inline(always)]
    fn sent_behind(&self, held: &HeldWord) -> SentCredit {
        SentCredit {
            requests: (held.0 & REQUEST_MASK) as u32,
            tokens: held.0 >> REQUEST_BITS,
            unsized_requests: u32::from(self.unsized_held.load(Ordering::Relaxed)),
        }
    }

    /// The estimates of the held dispatches that stated no size, as their
    /// word stands behind the held word's load ([`Self::estimated_held`]).
    #[inline(always)]
    fn estimated_behind(&self, _held: &HeldWord) -> u64 {
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
        let held_lanes = self.lock_lanes();
        let lanes = held_lanes.iter();
        lanes.flat_map(|lane| lane.0.iter().copied()).collect()
    }

    /// Whether the row holds a dispatch made at or before the sample: one
    /// a report sampled then reflects, which its release takes.
    #[inline(always)]
    pub(crate) fn reaches(&self, sampled_at_micros: u64) -> bool {
        self.oldest_micros.load(Ordering::Acquire) <= sampled_at_micros
    }

    /// The held token-work and requests, as the word stands.
    #[inline(always)]
    pub(crate) fn held_credit(&self) -> HeldCredit {
        Self::credit_of(self.held.load(Ordering::Relaxed))
    }

    /// The token-work and requests a held word says.
    #[inline(always)]
    pub(crate) const fn credit_of(held: u64) -> HeldCredit {
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
        // kept. Each of its dispatches stated its size, as long as that
        // count has not wrapped: it is as wide as the held word's count
        // of requests and counts no dispatch that word does not, so it
        // wraps only where that word's count has overflowed.
        if self.unsized_held.load(Ordering::Relaxed) != 0 {
            self.release_marked(held_lanes, sampled_at_micros);
        } else {
            for (at, dispatches) in held_lanes.iter_mut().enumerate() {
                while let Some(size) = dispatches.let_go(sampled_at_micros) {
                    self.take_out(Self::unpriced_stated(size));
                }
                self.fronts.publish(at, dispatches);
            }
        }
        let oldest = self
            .fronts
            .0
            .iter()
            .map(|front| front.load(Ordering::Acquire))
            .min()
            .unwrap_or(u64::MAX);
        self.oldest_micros.store(oldest, Ordering::Release);
    }

    /// The release of a row that holds a dispatch of no tokens: each one
    /// that goes leaves the estimates' word and that count first and the
    /// held word next. So a read adds no price of a dispatch the held
    /// word has let go, and may miss the price of each one that its held
    /// word counts and that goes between its own loads.
    #[cold]
    #[inline(never)]
    fn release_marked(
        &self,
        held_lanes: &mut [MutexGuard<'_, Dispatches>; DISPATCH_LANES],
        sampled_at_micros: u64,
    ) {
        for (at, dispatches) in held_lanes.iter_mut().enumerate() {
            while let Some(size) = dispatches.let_go(sampled_at_micros) {
                let gone = self.unprice(size);
                self.take_out(gone);
            }
            self.fronts.publish(at, dispatches);
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
    /// the one copied ([`Self::longest_read`]). The rows are compared by
    /// what was read under their locks, since the choice is handed the
    /// reads and no row: a release that crosses the copy takes no dispatch
    /// between the choice of a row and its read. The reads of a rank's
    /// rows are of different instants: a dispatch told between two of them
    /// is in the copy only where the longest read holds it. Each dispatch
    /// enters its row here as a send does ([`RankCredit::record`]), at its
    /// instant and its size: a row past its bound lets its oldest go. They
    /// enter oldest first, so each lands at its lane's end. What this
    /// tally held until then is let go, and it shares no row with the
    /// standing ones.
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
                let rows = std::iter::once(since).chain(later.filter_map(|tally| tally.get(rank)));
                let copy = RankCredit::default();
                for (at_micros, size) in Self::longest_read(rows.map(|row| row.held_sends())) {
                    copy.record(at_micros, size);
                }
                next.insert(*rank, Arc::new(copy));
            }
        }
        self.tally.store(Arc::new(next));
    }

    /// The longest of `reads`, oldest first: of two as long, the earlier.
    /// A read is one row's dispatches as its lanes hand them, lane by
    /// lane, so the longest is put in the order of its instants here.
    fn longest_read(reads: impl Iterator<Item = Vec<(u64, SendSize)>>) -> Vec<(u64, SendSize)> {
        let mut longest = reads.fold(Vec::new(), |longest, read| {
            if read.len() > longest.len() {
                read
            } else {
                longest
            }
        });
        longest.sort_unstable_by_key(|&(at_micros, _)| at_micros);
        longest
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

#[cfg(test)]
#[path = "../../tests/unit/tracking/rank_credit/tests.rs"]
pub(crate) mod tests;
