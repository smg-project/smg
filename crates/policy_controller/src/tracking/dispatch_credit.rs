//! Dispatches to a candidate since its last report, counted per profile.

use std::{
    marker::PhantomData,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use arc_swap::Guard;
use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    engine::{Candidate, LoadRow, RequestFacts, Target, WorkerId},
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

/// A profile's three settings on its count of sends, as its document
/// writes them: an unset one is `None`, or `false`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SentSettings {
    /// `sent_ttl_ms`: how long a send counts.
    pub(crate) ttl_ms: Option<u32>,
    /// `sent_report_delay_ms`: how long behind a send a report is sampled
    /// before it ends the send.
    pub(crate) report_delay_ms: Option<u32>,
    /// `sent_until_report_changes`: whether a report ends a send only
    /// where it differs from the report the count read of the row before
    /// it.
    pub(crate) until_report_changes: bool,
}

/// What a count that ends a send only at a changed report
/// (`sent_until_report_changes`) keeps of one row's reports: the report it
/// compared last, and the sample instant of the last report that differed
/// from one compared before it. It compares the reports its reads show
/// it, at each read of a row that holds a send: a report no read showed
/// it is compared with nothing. It tells one report from another by the
/// sample instant, as the engine's telling of reports does: a report
/// whose numbers change under an instant that stands is not compared
/// again. An entry makes one at such a count's first compare, so an entry
/// of any other count holds none.
#[derive(Debug)]
pub(crate) struct ComparedReports {
    /// The sample instant of the report compared last, `u64::MAX` where
    /// none was: a read that shows the same report takes no lock.
    seen_at: AtomicU64,
    /// The sample instant of the last report that differed from one
    /// compared before it, zero where none has.
    changed_at: AtomicU64,
    /// The newest report compared, its age and its source left out: what
    /// the host says of how the report reached it. With it, the sample
    /// instant it was first kept at: a repeat moves `seen_at` and leaves
    /// that instant. A report's first read holds it for the one compare.
    seen: Mutex<Option<(LoadRow, u64)>>,
}

impl Default for ComparedReports {
    fn default() -> Self {
        Self {
            seen_at: AtomicU64::new(u64::MAX),
            changed_at: AtomicU64::new(0),
            seen: Mutex::new(None),
        }
    }
}

impl ComparedReports {
    /// Compares `report`, sampled at `sample`, where it is not the report
    /// compared last, and says the instant the row's reports end its
    /// sends through: the sample of the last report that differed from
    /// one before it, or `sample` where the read shows an earlier report,
    /// and `None` where no report has differed. The first report compared
    /// has none before it and ends no send, and neither does one that
    /// repeats the report before it, whatever its age and its source.
    ///
    /// A read may show a report sampled before the one kept: a decision
    /// that holds an earlier snapshot. Where the two differ, the instant
    /// is the later of the late report's sample and the sample the kept
    /// report was first kept at: the kept report differed from a report
    /// before it there, and its repeats since differ from nothing. So a
    /// late report sampled between two reads of the kept one dates the
    /// change at its own sample, and no send made behind it ends early.
    /// The instant only rises.
    ///
    /// `holds` says whether the row holds a send the read's instant
    /// reaches, by the lanes' own fronts: asked ahead of a write, so a
    /// read that reaches none keeps no report.
    fn changed_through(
        &self,
        report: &LoadRow,
        sample: u64,
        holds: impl FnOnce() -> bool,
    ) -> Option<u64> {
        if self.seen_at.load(Ordering::Acquire) != sample && holds() {
            let rest = LoadRow {
                report_age_ms: 0,
                source: crate::engine::SignalSource::None,
                ..*report
            };
            let mut seen = self.seen.lock();
            // Read again under the lock: another read may have compared
            // this report, or a newer one, meanwhile.
            let seen_at = self.seen_at.load(Ordering::Acquire);
            let differs = seen.filter(|&(other, _)| sample != seen_at && other != rest);
            if let Some((_, since)) = differs {
                self.changed_at
                    .fetch_max(sample.max(since), Ordering::AcqRel);
            }
            if seen.is_none() || sample > seen_at {
                // A repeat keeps the instant its report was first kept at.
                let since = seen.filter(|_| differs.is_none());
                *seen = Some((rest, since.map_or(sample, |(_, since)| since)));
                self.seen_at.store(sample, Ordering::Release);
            }
        }
        let changed = self.changed_at.load(Ordering::Acquire).min(sample);
        (changed != 0).then_some(changed)
    }
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
/// count under the same settings, or else of the one that a profile
/// of its maker's name made, or else of one that no count of the new
/// document took, and shares them with it
/// ([`Engine::succeed`](crate::engine::Engine::succeed)). Each count reads them
/// under its own three settings, and a send that one
/// read releases has left the other count too: of two report delays, a
/// read under the shorter ends a send for both counts, and a read of a
/// count that does not set `sent_until_report_changes` ends a send for
/// one that sets it. So while decisions of the document before still
/// run, a report that repeats the one before it can end a send.
///
/// Two stages of a decision read under the same row sample and clock. With
/// the same membership, their reads differ only if a send lands or another
/// decision releases sends between them. The later read has the newer count.
///
/// A report sampled at or after a send ends it, as do the profile's time
/// limit and the row's removal. Under a report delay the report that ends
/// a send is one sampled that long after the send, or longer: a report
/// sampled sooner behind it leaves it counted. Under
/// `sent_until_report_changes` the report that ends a send also differs
/// from the report the count read of the row before it, by anything but its
/// age and its source: a report that repeats the one before it ends no
/// send, and neither does the first the count reads of a row. The count
/// compares at each read of a row that holds a send, so every such read
/// takes the release's path. Under a request clock of zero it compares only
/// where the row's sample, less the delay, reaches a send: a report is then
/// compared with the last one such a read showed. When a send arrives with
/// 8,192 already held on a row, the tally lets the row's oldest go to bound
/// the count. This threshold applies at any age.
#[derive(Debug)]
pub struct DispatchCredit {
    pub(crate) credits: Credits,
    /// The profile's time limit, `u64::MAX` where it sets none: no clock
    /// reaches a cutoff that far back. Zero under a count that ends a
    /// send only at a changed report: a pass's cutoff is then its
    /// request's clock, so a read reaches every row that holds a send and
    /// compares the row's report where it releases ([`Self::release`]),
    /// and a row that holds none is read as under any other count.
    ttl_micros: u64,
    /// The time limit of a count that ends a send only at a changed
    /// report, where its profile sets one. No other count reads it.
    changed_ttl_micros: Option<u64>,
    /// The profile's report delay: a report ends the sends made this long
    /// before its sample, or longer. Zero where the profile sets none: a
    /// report then ends the sends made at its sample and before.
    report_delay_micros: u64,
    /// Whether the profile sets either of the two, or ends its sends
    /// only at a changed report: what a pass asks once
    /// ([`Self::limited`]).
    limited: bool,
    /// What a read of a row's totals runs where its instant reaches a
    /// send of the row: the row's own release, or the compare of a count
    /// that ends a send only at a changed report. Picked once, where the
    /// document's count is made: no read tests the count's kind.
    release: fn(&RankCredit, u64, &Candidate, &DispatchCreditPass<'_>),
    /// The instant a list of a row's sends ends them through, where its
    /// own instant reaches one: that instant, or the one the row's
    /// reports give, and `None` where nothing ends a send. Picked with
    /// [`Self::release`].
    listed: fn(&RankCredit, u64, &Candidate, &DispatchCreditPass<'_>) -> Option<u64>,
    /// The settings as the document writes them: a later profile of the
    /// document that writes the same three reads this count.
    pub(crate) settings: SentSettings,
    /// The name of the profile that made the count, the first of its
    /// readers: a later document's count that finds no standing count
    /// under its settings is matched by this name
    /// ([`Engine::succeed`](crate::engine::Engine::succeed)).
    pub(crate) profile: String,
}

impl DispatchCredit {
    pub(crate) fn new(profile: &str, settings: SentSettings) -> Self {
        let micros = |ms: u32| u64::from(ms) * 1000;
        let ttl_micros = settings.ttl_ms.map_or(u64::MAX, micros);
        let changed = settings.until_report_changes;
        Self {
            credits: Credits::default(),
            ttl_micros: if changed { 0 } else { ttl_micros },
            changed_ttl_micros: settings.ttl_ms.map(micros),
            report_delay_micros: settings.report_delay_ms.map_or(0, micros),
            limited: settings.ttl_ms.is_some() || settings.report_delay_ms.is_some() || changed,
            release: if changed {
                Self::release_changed
            } else {
                Self::release_reported
            },
            listed: if changed {
                Self::changed_through
            } else {
                |_, through, _, _| Some(through)
            },
            settings,
            profile: profile.to_owned(),
        }
    }

    /// [`Self::release`] of a count that sets no rule on a changed
    /// report: the row's own release through the read's instant.
    fn release_reported(
        since: &RankCredit,
        through: u64,
        _: &Candidate,
        _: &DispatchCreditPass<'_>,
    ) {
        since.release_through(through);
    }

    /// [`Self::release`] of a count that ends a send only at a changed
    /// report: the row's release through the instant its reports give
    /// ([`Self::changed_through`]), where they give one.
    fn release_changed(
        since: &RankCredit,
        through: u64,
        row: &Candidate,
        pass: &DispatchCreditPass<'_>,
    ) {
        let ends = Self::changed_through(since, through, row, pass);
        if let Some(ends) = ends.filter(|&ends| since.reaches(ends)) {
            since.release_through(ends);
        }
    }

    /// The instant a read of `row` ends its sends through under a count
    /// that ends a send only at a changed report, where the read's own
    /// instant is `through`: the sample of the row's last report that
    /// differed from one before it ([`ComparedReports::changed_through`])
    /// less the count's report delay, or the request's clock less the
    /// count's time limit where that is later. The limit runs by the
    /// clock alone, which such a count's pass keeps as its cutoff, and a
    /// clock of zero applies none. `None`
    /// where neither gives an instant: no report of the row has differed
    /// the delay behind any instant, and the limit has not run. Such a
    /// read ends no send, one made at instant zero included. A row
    /// without a report ends its sends by the limit alone.
    fn changed_through(
        since: &RankCredit,
        through: u64,
        row: &Candidate,
        pass: &DispatchCreditPass<'_>,
    ) -> Option<u64> {
        let credit = pass.credit;
        let changed = row.load.as_ref().and_then(|report| {
            since.reports.get_or_init(Box::default).changed_through(
                report,
                row.load_sampled_at_micros,
                || since.holds_through(through),
            )
        });
        let limit = credit
            .changed_ttl_micros
            .filter(|_| pass.cutoff != 0)
            .and_then(|limit| pass.cutoff.checked_sub(limit));
        changed
            .and_then(|at| at.checked_sub(credit.report_delay_micros))
            .max(limit)
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
    /// sends, or ends them only at a changed report. The pass a stage's
    /// row loop makes over a count with none of the three
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
            credit: self,
        }
    }

    /// [`Self::pass`] for a count that sets no time limit and no report
    /// delay and ends its sends at any report ([`Self::limited`]): the
    /// pass holds the tally map alone, and neither a cutoff nor a delay
    /// is worked out or kept.
    #[inline(always)]
    pub(crate) fn reads_unlimited(&self) -> UnlimitedPass<'_> {
        debug_assert!(
            !self.limited(),
            "the count sets no limit, no delay and no compare"
        );
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
    credit: &'a DispatchCredit,
}

impl DispatchCreditPass<'_> {
    /// The instant a row's read releases through, where the row's report
    /// was sampled at `sample`: the sample less the count's report delay,
    /// or the time cutoff where that is later. Every read of the count
    /// takes the delay off here. A sample sooner than the delay behind
    /// the clock's start releases through instant zero, as a row without
    /// a report does. A count that ends its sends only at a changed
    /// report takes the delay off the instant of that report too
    /// ([`DispatchCredit::changed_through`]), and there an instant sooner
    /// than the delay behind the clock's start ends no send.
    #[inline(always)]
    fn through(&self, sample: u64) -> u64 {
        sample.saturating_sub(self.delay).max(self.cutoff)
    }

    /// The count's release of a row whose sends the read's instant
    /// reaches ([`DispatchCredit::release`]). A method, so that a read's
    /// closure keeps the pass alone and loads nothing of it for a row
    /// that releases none.
    #[inline(always)]
    fn release(&self, since: &RankCredit, through: u64, row: &Candidate) {
        (self.credit.release)(since, through, row, self);
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
                let through = self.through(sample);
                let sent = since.sent_released(through, |since| {
                    self.release(since, through, row);
                });
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
                let through = self.through(sample);
                since.counts_released(through, |since| {
                    self.release(since, through, row);
                })
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
            let through = self.through(sample);
            since.each_released(
                through,
                |since| (self.credit.listed)(since, through, row, self),
                each,
            );
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
        debug_assert!(row.load.is_some(), "the row holds a report");
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(SentCredit::default, |since| {
                let through = self.through(row.load_sampled_at_micros);
                since.sent_released(through, |since| {
                    self.release(since, through, row);
                })
            })
    }
}

impl ReportedSends for UnlimitedPass<'_> {
    #[inline(always)]
    fn reported(&self, row: &Candidate) -> SentCredit {
        debug_assert!(row.load.is_some(), "the row holds a report");
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(SentCredit::default, |since| {
                since.sent_through(row.load_sampled_at_micros)
            })
    }
}

#[cfg(test)]
mod tests;
