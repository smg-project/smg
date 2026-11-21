//! Dispatches to a candidate since its last report, counted per profile.

use std::{marker::PhantomData, sync::Arc};

use arc_swap::Guard;
use rustc_hash::FxHashMap;

use crate::{
    stages::builtins::{Credits, SinceReport},
    Candidate, Lease, Membership, RequestFacts, Side, Target, WorkerId,
};

/// What the deciding host sent a row that no report of the row reflects.
/// The type is open to further numbers: code outside the crate reads its
/// fields, and matches them with `..`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct Sent {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens the sends stated. A send of unstated size adds none.
    pub tokens: u64,
    /// How many of the sends stated no size. A reader that prices such a
    /// send chooses the price: `least-load` takes its mean prefill.
    pub unsized_requests: u32,
}

/// The sends a row holds and the prompt tokens they stated: the two
/// numbers a document's `sent.` terms read, and what a stage reads that
/// prices no send of unstated size ([`SentReads::counts`]). Such a send is
/// one of `requests` and adds no tokens; [`Sent::unsized_requests`] says
/// how many there are. The pair is closed: a further number goes on
/// [`Sent`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SentCounts {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens the sends stated. A send of unstated size adds none.
    pub tokens: u64,
}

/// A profile's count of what its host sent each row since the row's last
/// report, under the profile's time limit. The engine feeds it and stages
/// read it: a registered stage, `least-cost` through its `sent.` terms,
/// and `least-load` for its credit. Under a standalone controller it
/// counts sends from every gateway through that controller.
///
/// A new document's count takes over the rows' entries of the standing
/// profile of the same name and shares them with it
/// ([`Engine::succeed`](crate::Engine::succeed)). Each count reads them
/// under its own time limit, and a send that one read releases has left
/// the other count too.
///
/// Two stages of a decision read under the same row sample and clock. With
/// the same membership, their reads differ only if a send lands or another
/// decision releases sends between them. The later read has the newer count.
///
/// A report sampled at or after a send ends it, as do the profile's time
/// limit and the row's removal. When a send arrives with 8,192 already
/// held on a row, the tally retires the oldest batch to bound the count.
/// This threshold applies at any age.
#[derive(Debug)]
pub struct SentCredit {
    pub(crate) credits: Credits,
    /// The profile's time limit, `u64::MAX` where it sets none: no clock
    /// reaches a cutoff that far back.
    ttl_micros: u64,
    /// The profile's name: the count of a later document's profile of
    /// this name takes over this one's entries.
    pub(crate) profile: String,
}

impl SentCredit {
    pub(crate) fn new(profile: &str, ttl_ms: Option<u32>) -> Self {
        Self {
            credits: Credits::default(),
            ttl_micros: ttl_ms.map_or(u64::MAX, |ms| u64::from(ms) * 1000),
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
            |since, at_micros, tokens| since.record(at_micros, tokens),
        );
    }

    pub(crate) fn membership(&self, change: &Membership<'_>) {
        self.credits.membership(change, |_| {});
    }

    /// Whether the profile sets a time limit on its sends. A pass over a
    /// count with none compares no cutoff.
    #[inline(always)]
    pub(crate) fn limited(&self) -> bool {
        self.ttl_micros != u64::MAX
    }

    /// One row's sends.
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> Sent {
        self.reads(facts).of(row)
    }

    /// One pass's reads. The pass loads the tally map once and reads each
    /// row's count when asked. A request clock of zero applies no time limit.
    #[inline]
    pub fn reads<'a>(&'a self, facts: &RequestFacts<'_>) -> SentReads<'a> {
        SentReads {
            tally: self.credits.tally.load(),
            cutoff: facts.now_micros.saturating_sub(self.ttl_micros),
            credit: PhantomData,
        }
    }

    /// [`Self::reads`] for a count that sets no time limit
    /// ([`Self::limited`]): the pass has no cutoff, and none is worked
    /// out.
    #[inline(always)]
    pub(crate) fn reads_unlimited(&self) -> SentReads<'_> {
        debug_assert!(!self.limited(), "the count sets no time limit");
        SentReads {
            tally: self.credits.tally.load(),
            cutoff: 0,
            credit: PhantomData,
        }
    }
}

type Tally = FxHashMap<(WorkerId, u16), Arc<SinceReport>>;

/// A pass over a profile's send tally at one request's clock.
pub struct SentReads<'a> {
    tally: Guard<Arc<Tally>>,
    cutoff: u64,
    credit: PhantomData<&'a SentCredit>,
}

impl SentReads<'_> {
    /// Whether the count holds an entry of any row. A row enters at its
    /// first send and leaves with its removal, so a count that holds none
    /// reads zero of every row: a pass that only sums its reads skips
    /// them.
    #[inline(always)]
    pub(crate) fn holds_any(&self) -> bool {
        !self.tally.is_empty()
    }

    /// The sends since the later of the row's sample and the time cutoff.
    /// A row without a report has a sample instant of zero.
    ///
    /// Inlined into its caller's row loop, so the three numbers come back
    /// in no memory. Where the row holds sends it loads three words, read
    /// or not: a stage that wants the requests and the stated tokens alone
    /// reads [`Self::counts`], which loads two.
    #[inline(always)]
    pub fn of(&self, row: &Candidate) -> Sent {
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(Sent::default, |since| {
                let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
                let sent = since.sent_through(sample.max(self.cutoff));
                // The count of sends of no stated size is read apart from
                // the other two: beside a send that lands between the
                // reads it is held to the sends counted.
                Sent {
                    unsized_requests: sent.unsized_requests.min(sent.requests),
                    ..sent
                }
            })
    }

    /// The requests and the stated tokens of [`Self::of`], for a stage
    /// that prices no send of unstated size. It releases what a report or
    /// the time limit reaches, as [`Self::of`] does, and loads two words
    /// where it releases none: the third number's word is read only by a
    /// release. Inlined into its caller's row loop.
    #[inline(always)]
    pub fn counts(&self, row: &Candidate) -> SentCounts {
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(SentCounts::default, |since| {
                let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
                since.credit_through(sample.max(self.cutoff)).counts()
            })
    }

    /// [`Self::of`] for a row that holds a report, in a stage's row loop.
    /// `TIMED` is whether the count sets a time limit
    /// ([`SentCredit::limited`]), settled once a pass: a pass over a count
    /// with none compares no cutoff. The count of sends of no stated size
    /// is as its word stands: beside a send that lands between the reads
    /// it may differ by that send.
    #[inline(always)]
    pub(crate) fn reported<const TIMED: bool>(&self, row: &Candidate) -> Sent {
        debug_assert!(row.load.is_some(), "the row holds a report");
        debug_assert!(
            TIMED || self.cutoff == 0,
            "a count with no limit has no cutoff"
        );
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(Sent::default, |since| {
                let sample = row.load_sampled_at_micros;
                since.sent_through(if TIMED {
                    sample.max(self.cutoff)
                } else {
                    sample
                })
            })
    }
}

#[cfg(test)]
mod tests;
