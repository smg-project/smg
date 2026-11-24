//! Dispatches to a candidate since its last report, counted per profile.

use std::{marker::PhantomData, sync::Arc};

use arc_swap::Guard;
use rustc_hash::FxHashMap;

use crate::{
    engine::{Candidate, RequestFacts, Target, WorkerId},
    stages::scorers::{Credits, RankCredit, SentCredit},
    tracking::{Lease, Membership, Side},
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
    /// ([`DispatchCounts::tokens`]). Such a reader asks for each request's
    /// size ([`Needs::prompt_size`](crate::document::Needs::prompt_size)):
    /// a host that can send a request on without reading its prompt states
    /// a text's bytes only where a stage asks.
    pub estimated_tokens: u64,
}

/// The sends a row holds and their prompt tokens: the two numbers a
/// document's `sent.` terms read ([`DispatchCreditPass::counts`]). A send
/// that stated no token count is one of `requests` and adds its estimate,
/// the bytes of its text over four: [`CreditedDispatches`] states the
/// stated tokens and the estimates apart, and how many such sends there
/// are. The pair is closed: a further number goes on
/// [`CreditedDispatches`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DispatchCounts {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens of the sends: the count each one stated, and for
    /// a send that stated none the bytes of its text over four. A send of
    /// no tokens and no text adds none. A stage that reads this number
    /// asks for each request's size
    /// ([`Needs::prompt_size`](crate::document::Needs::prompt_size)), as
    /// the document's two stages do where a term names `sent.tokens`.
    pub tokens: u64,
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
/// delay read one count: the engine records a send in every count, and
/// only a decision's read lets one go, so a count of its own would keep
/// the sends of a profile that decides no request.
///
/// A new document's count takes over the rows' entries of the standing
/// count that a profile of the same name made, and shares them with it
/// ([`Engine::succeed`](crate::engine::Engine::succeed)). Each count reads them
/// under its own time limit and its own report delay, and a send that one
/// read releases has left the other count too: of two report delays, a
/// read under the shorter ends a send for both counts.
///
/// Two stages of a decision read under the same row sample and clock. With
/// the same membership, their reads differ only if a send lands or another
/// decision releases sends between them. The later read has the newer count.
///
/// A report sampled at or after a send ends it, as do the profile's time
/// limit and the row's removal. Under a report delay the report that ends
/// a send is one sampled that long after the send, or longer: a report
/// sampled sooner behind it leaves it counted. When a send arrives with
/// 8,192 already held on a row, the tally lets the row's oldest go to
/// bound the count. This threshold applies at any age.
#[derive(Debug)]
pub struct DispatchCredit {
    pub(crate) credits: Credits,
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
    /// the document that writes the same two reads this count.
    pub(crate) settings: SentSettings,
    /// The name of the profile that made the count, the first of its
    /// readers: the count a later document's profile of this name makes
    /// takes over this one's entries.
    pub(crate) profile: String,
}

impl DispatchCredit {
    pub(crate) fn new(profile: &str, settings: SentSettings) -> Self {
        let micros = |ms: u32| u64::from(ms) * 1000;
        Self {
            credits: Credits::default(),
            ttl_micros: settings.ttl_ms.map_or(u64::MAX, micros),
            report_delay_micros: settings.report_delay_ms.map_or(0, micros),
            limited: settings.ttl_ms.is_some() || settings.report_delay_ms.is_some(),
            settings,
            profile: profile.to_owned(),
        }
    }

    /// Takes over the rows' entries `standing` holds: the map as it
    /// stands, so both counts credit and release the same entries, and
    /// each enters a rank the other never saw in a map of its own.
    pub(crate) fn succeed(&self, standing: &Self) {
        self.credits.tally.store(standing.credits.tally.load_full());
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
        self.credits.dispatched(
            lease,
            target,
            side,
            since_decision_micros,
            present,
            |since, at_micros, size| since.record(at_micros, size),
        );
    }

    pub(crate) fn membership(&self, change: &Membership<'_>) {
        self.credits.membership(change, |_| {});
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
            tally: self.credits.tally.load(),
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
            tally: self.credits.tally.load(),
            credit: PhantomData,
        }
    }
}

type Tally = FxHashMap<(WorkerId, u16), Arc<RankCredit>>;

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
mod tests;
