//! Dispatches to a candidate since its last report, counted per profile.

use std::{
    marker::PhantomData,
    sync::{Arc, OnceLock},
};

use arc_swap::Guard;

use crate::{
    engine::{Candidate, RequestFacts, Target},
    tracking::{
        rank_credit::{Credits, DispatchCounts, RankTable, SentCredit},
        Lease, Membership, Side,
    },
};

/// What the deciding host sent a row that no report of the row reflects.
/// The type is open to further numbers: code outside the crate reads its
/// fields, and matches them with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct CreditedDispatches {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens the sends stated. A send of unstated size adds none.
    pub tokens: u64,
    /// How many of the sends stated no size. A reader that prices such a
    /// send chooses the price: `least-load` takes its mean prefill.
    pub unsized_requests: u32,
    /// The estimates of the sends that stated no token count, summed: for
    /// each, the bytes of its text over four, and nothing for a send of
    /// no text. A reader that prices such a send by its text adds this to
    /// `tokens`, as a document's `sent.` terms do
    /// ([`DispatchCounts::tokens`]). Such a reader is to ask for each
    /// request's size
    /// ([`Needs::prompt_size`](crate::document::Needs::prompt_size)):
    /// where no stage asks, a host that can send a request on without
    /// reading its prompt may state no bytes of its text.
    pub estimated_tokens: u64,
}

/// A profile's two settings on its count of sends, as its document writes
/// them: an unset one is `None`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SentSettings {
    /// `sent_ttl_ms`: how long a send counts.
    pub(crate) ttl_ms: Option<u32>,
    /// `sent_report_delay_ms`: how long behind a send a report is sampled
    /// before it ends the send.
    pub(crate) report_delay_ms: Option<u32>,
}

/// A profile's count of what its host sent each row since the row's last
/// report, under the profile's time limit and its report delay. The engine
/// feeds it and stages read it: a registered stage, `least-cost` through
/// its `sent.` terms, and `least-load` for its credit. Under a standalone
/// controller it counts sends from every gateway through that controller.
/// The profiles of a document that set the same time limit and report
/// delay read one count: the engine tells every count each send it
/// counts, and a report or a time limit ends a send only at a read. A
/// profile that decides no request would give a count of its own no read
/// from its decisions.
///
/// Where a new document keeps a profile that reads a count, its engine
/// goes on with that count, and the count built for the document's
/// readers follows it (`follow`): the readers kept and the readers
/// built new read one set of entries. Any other count of a new document
/// takes over the rows' entries of the standing count under the same two
/// settings and shares them with it
/// ([`Engine::succeed`](crate::engine::Engine::succeed)): a send that a
/// read of one releases has left the other count too. A count under two
/// settings no standing count has shares none. A standing count has let
/// go what its own settings ended, so such a count takes a copy of the
/// sends that stand, of each row those of the standing count that holds
/// the most, and its own reads let go what its own settings end.
///
/// Two stages of a decision read under the same row sample and clock. With
/// the same membership, their reads can still differ: a send can land
/// between them, a send to a full row can let one go, and another read
/// can release sends. The later read has the newer count.
///
/// A report sampled at or after a send ends it, as do the profile's time
/// limit and the row's removal. Under a report delay the report that ends
/// a send is one sampled that long after the send, or longer: a report
/// sampled sooner behind it leaves it counted. When a send arrives with
/// 8,192 or more held on a row, the tally lets one of the row's sends go
/// to bound the count, or none: as a rule the oldest, at any age. None
/// goes where the row, the new send counted, holds no more than 8,192.
#[derive(Debug)]
pub struct DispatchCredit {
    /// The rows' entries of this count's own: what every word of it reads
    /// and writes, unless it follows another count ([`Self::credits`]).
    credits: Credits,
    /// The standing count this one follows, from the keep that built it
    /// beside that count's readers on ([`Self::follow`]): every word of
    /// this count reads and writes that count's entries. Unset for a count
    /// an engine holds as its own.
    follows: OnceLock<Arc<DispatchCredit>>,
    /// The profile's time limit, `u64::MAX` where it sets none: no clock
    /// reaches a cutoff that far back.
    ttl_micros: u64,
    /// The profile's report delay: a report ends the sends made this long
    /// before its sample, or longer. Zero where the profile sets none: a
    /// report then ends the sends made at its sample and before.
    report_delay_micros: u64,
    /// Whether the profile sets either of the two: what a pass asks once
    /// ([`Self::limited`]).
    limited: bool,
    /// The two settings as the document writes them: a later profile of
    /// the document that writes the same two reads this count, and a
    /// later document's count under the same two takes its entries over
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)).
    pub(crate) settings: SentSettings,
}

impl DispatchCredit {
    pub(crate) fn new(settings: SentSettings) -> Self {
        let micros = |ms: u32| u64::from(ms) * 1000;
        Self {
            credits: Credits::default(),
            follows: OnceLock::new(),
            ttl_micros: settings.ttl_ms.map_or(u64::MAX, micros),
            report_delay_micros: settings.report_delay_ms.map_or(0, micros),
            limited: settings.ttl_ms.is_some() || settings.report_delay_ms.is_some(),
            settings,
        }
    }

    /// The rows' entries every word of this count reads and writes: its
    /// own, or those of the count it follows. One load of the cell, which
    /// a count that follows none answers by its own entries.
    #[inline(always)]
    pub(crate) fn credits(&self) -> &Credits {
        match self.follows.get() {
            Some(leader) => &leader.credits,
            None => &self.credits,
        }
    }

    /// This count reads and writes `leader`'s entries from here on. A new
    /// document's count is built for its readers, and where the document
    /// keeps a standing reader of the same two settings the engine goes on
    /// with that reader's count: the count built follows it, so the
    /// readers built new and the reader kept read one set of entries, and
    /// a rank first heard later stands once. The engine tells the count it
    /// goes on with and no count that follows: a send is counted once.
    /// Called once, before a word reaches this count, with a count that
    /// follows none: a count follows one count for its life, and no chain
    /// of them grows.
    #[cfg(feature = "in-process")]
    pub(crate) fn follow(&self, leader: &Arc<Self>) {
        debug_assert_eq!(self.settings, leader.settings, "one pair of settings");
        debug_assert!(leader.follows.get().is_none(), "its leader follows none");
        debug_assert!(self.credits.tally.load().is_empty(), "no word yet");
        let set = self.follows.set(Arc::clone(leader));
        debug_assert!(set.is_ok(), "a count follows one count");
    }

    /// Takes over the rows' entries `standing` holds, a count under the
    /// same two settings: the map as it stands, so both counts credit and
    /// release the same entries, and each enters a rank the other never
    /// saw in a map of its own. Neither count follows another: an engine
    /// holds the count followed in the place of each count that follows,
    /// and it alone calls this. A count an engine kept is the standing
    /// one itself and takes nothing over: stored over itself, its map
    /// would lose a rank another thread entered meanwhile.
    pub(crate) fn succeed(&self, standing: &Self) {
        debug_assert!(!std::ptr::eq(self, standing), "a count kept takes none");
        debug_assert_eq!(self.settings, standing.settings, "one pair of settings");
        debug_assert!(self.follows.get().is_none(), "an engine's own count");
        debug_assert!(standing.follows.get().is_none(), "an engine's own count");
        let held = standing.credits.tally.load_full();
        self.credits.tally.store(held);
    }

    /// Takes a copy of the sends the `standing` counts hold, none of them
    /// under this count's two settings: of each row, the sends of the
    /// standing count that holds the most there, each at its instant and
    /// its size ([`Credits::copy_fullest`]). The count shares no entry
    /// with them: a read of it releases its own alone, and it starts
    /// empty where no count stands. An engine alone calls this, with its
    /// own count and a standing engine's own.
    pub(crate) fn copy_fullest(&self, standing: &[Arc<Self>]) {
        debug_assert!(self.follows.get().is_none(), "an engine's own count");
        let standing = standing.iter().map(|count| {
            debug_assert!(count.follows.get().is_none(), "an engine's own count");
            &count.credits
        });
        self.credits.copy_fullest(standing);
    }

    /// Counts one send. `present` is asked only where the send would
    /// enter a rank the count does not know.
    pub(crate) fn dispatched(
        &self,
        lease: &Lease,
        target: &Target,
        side: Side,
        since_decision_micros: u32,
        present: impl FnOnce() -> bool,
    ) {
        self.credits().dispatched(
            lease,
            target,
            side,
            since_decision_micros,
            present,
            |since, at_micros, size| since.record(at_micros, size),
        );
    }

    pub(crate) fn membership(&self, change: &Membership<'_>) {
        self.credits().membership(change);
    }

    /// Whether the profile sets a time limit or a report delay on its
    /// sends. The pass a stage's row loop makes over a count with neither
    /// ([`Self::reads_unlimited`]) compares no cutoff and takes no delay
    /// off a sample.
    #[inline(always)]
    pub(crate) fn limited(&self) -> bool {
        self.limited
    }

    /// One row's sends.
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> CreditedDispatches {
        self.pass(facts).of(row)
    }

    /// One pass's reads. The pass loads the tally map once and reads each
    /// row's count when asked. A request clock of zero applies no time limit.
    #[inline]
    pub fn pass<'a>(&'a self, facts: &RequestFacts<'_>) -> DispatchCreditPass<'a> {
        DispatchCreditPass {
            tally: self.credits().tally.load(),
            cutoff: facts.now_micros.saturating_sub(self.ttl_micros),
            delay: self.report_delay_micros,
            credit: PhantomData,
        }
    }

    /// [`Self::pass`] for a count that sets no time limit and no report
    /// delay ([`Self::limited`]): the pass holds the tally map alone, and
    /// neither a cutoff nor a delay is worked out or kept.
    #[inline(always)]
    pub(crate) fn reads_unlimited(&self) -> UnlimitedPass<'_> {
        debug_assert!(!self.limited(), "the count sets no limit and no delay");
        UnlimitedPass {
            tally: self.credits().tally.load(),
            credit: PhantomData,
        }
    }
}

type Tally = RankTable;

/// A pass over a profile's send tally at one request's clock.
pub struct DispatchCreditPass<'a> {
    tally: Guard<Arc<Tally>>,
    cutoff: u64,
    /// The count's report delay, zero where its profile sets none.
    delay: u64,
    credit: PhantomData<&'a DispatchCredit>,
}

impl DispatchCreditPass<'_> {
    /// The instant a row's read releases through, where the row's report
    /// was sampled at `sample`: the sample less the count's report delay,
    /// or the time cutoff where that is later. The one place the delay is
    /// applied, so every reader of the count follows it. A sample sooner
    /// than the delay behind the clock's start releases through instant
    /// zero, as a row without a report does.
    #[inline(always)]
    fn through(&self, sample: u64) -> u64 {
        sample.saturating_sub(self.delay).max(self.cutoff)
    }

    /// Whether the count holds an entry of any row. A row enters at its
    /// first send and leaves with its removal, so a count that holds none
    /// reads zero of every row: a pass that only sums its reads skips
    /// them.
    #[inline(always)]
    pub(crate) fn holds_any(&self) -> bool {
        !self.tally.is_empty()
    }

    /// The sends since the later of the row's sample, less the count's
    /// report delay, and the time cutoff. A row without a report has a
    /// sample instant of zero.
    ///
    /// Inlined into its caller's row loop, so the four numbers come back
    /// in no memory. Where the row holds sends it loads four words, read
    /// or not: a stage that wants the requests and their tokens, stated
    /// or estimated, reads [`Self::counts`], which loads three.
    #[inline(always)]
    pub fn of(&self, row: &Candidate) -> CreditedDispatches {
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(CreditedDispatches::default, |since| {
                let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
                let sent = since.sent_through(self.through(sample));
                // The count of sends of no stated size and their
                // estimates are read apart from the other two: beside a
                // send that lands between the reads the count is held to
                // the sends counted.
                CreditedDispatches {
                    requests: sent.requests,
                    tokens: sent.tokens,
                    unsized_requests: sent.unsized_requests.min(sent.requests),
                    estimated_tokens: since.estimated(),
                }
            })
    }

    /// The requests of [`Self::of`] and their tokens, stated or estimated:
    /// what a document's `sent.` terms read. A send that stated no token
    /// count adds the bytes of its text over four. It releases what a
    /// report or the time limit reaches, as [`Self::of`] does, and loads
    /// three words where it releases none: the count of the sends of no
    /// stated size is read only by a release. Inlined into its caller's
    /// row loop.
    #[inline(always)]
    pub fn counts(&self, row: &Candidate) -> DispatchCounts {
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(DispatchCounts::default, |since| {
                let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
                since.counts_through(self.through(sample))
            })
    }

    /// Calls `each` with the instant and the stated prompt tokens of every
    /// send of the row that still counts: the sends [`Self::of`] counts,
    /// one call a send. The instant is the one the count dates the send
    /// by: its lease's issue instant plus the time its host reported from
    /// the decision to the dispatch, in microseconds of the engine's
    /// clock. A send of no stated size states no tokens. The sends come
    /// lane by lane as the count keeps them, oldest first within a lane: a
    /// stage that wants one order sorts.
    ///
    /// It releases what a report or the time limit reaches, as
    /// [`Self::of`] does, and holds the row's sends still from that
    /// release until it returns: a send to the row waits and lands behind
    /// the read. So `each` must not read this row through any count that
    /// holds it, report a send or wait, because each may need what the
    /// read holds. A document's count that took over the rows of the one
    /// it follows shares them with it. A stage that asks takes the row's
    /// locks at every read, which a total takes only where it releases.
    pub fn sends(&self, row: &Candidate, each: impl FnMut(u64, u32)) {
        if let Some(since) = self.tally.get(&(row.id, row.dp_rank)) {
            let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
            since.each_through(self.through(sample), each);
        }
    }
}

/// A pass over a count that sets no time limit and no report delay
/// ([`DispatchCredit::reads_unlimited`]). A row's report alone ends its
/// sends there, so the pass keeps the tally map and nothing else: a profile
/// that sets neither setting stores no cutoff and no delay.
pub(crate) struct UnlimitedPass<'a> {
    tally: Guard<Arc<Tally>>,
    credit: PhantomData<&'a DispatchCredit>,
}

/// The requests, the stated tokens and the count of the sends of no stated
/// size of [`DispatchCreditPass::of`], for a row that holds a report, in
/// the row loop of a stage that prices such a send itself: three loads,
/// and no estimate. Each of a count's two passes answers it by its own
/// rule, and which pass reads is settled once a pass
/// ([`DispatchCredit::limited`]): the pass of a count with no time limit
/// and no report delay compares no cutoff and takes no delay off a sample.
/// The count of sends of no stated size is as its word stands: beside a
/// send that lands between the reads it may differ by that send.
pub(crate) trait ReportedSends {
    fn reported(&self, row: &Candidate) -> SentCredit;
}

impl ReportedSends for DispatchCreditPass<'_> {
    #[inline(always)]
    fn reported(&self, row: &Candidate) -> SentCredit {
        reported_through(&self.tally, row, |sample| self.through(sample))
    }
}

impl ReportedSends for UnlimitedPass<'_> {
    #[inline(always)]
    fn reported(&self, row: &Candidate) -> SentCredit {
        reported_through(&self.tally, row, |sample| sample)
    }
}

/// The sends of a row that holds a report, released through the instant
/// `through` makes of the report's sample.
#[inline(always)]
fn reported_through(
    tally: &Tally,
    row: &Candidate,
    through: impl FnOnce(u64) -> u64,
) -> SentCredit {
    debug_assert!(row.load.is_some(), "the row holds a report");
    tally
        .get(&(row.id, row.dp_rank))
        .map_or_else(SentCredit::default, |since| {
            since.sent_through(through(row.load_sampled_at_micros))
        })
}

#[cfg(test)]
#[path = "../../tests/unit/tracking/dispatch_credit/tests.rs"]
mod tests;
