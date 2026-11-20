//! Dispatches to a candidate since its last report, counted per profile.

use std::{marker::PhantomData, sync::Arc};

use arc_swap::Guard;
use rustc_hash::FxHashMap;

use crate::{
    builtins::{Credits, SinceReport},
    Candidate, Lease, Membership, RequestFacts, Side, Target, WorkerId,
};

/// What the deciding host sent a row that no report of the row reflects.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sent {
    /// The number of sends.
    pub requests: u32,
    /// The prompt tokens the sends stated. A send of unstated size adds none.
    pub tokens: u64,
}

/// A profile's count of what its host sent each row since the row's last
/// report, under the profile's time limit. The engine feeds it and stages
/// read it. Under a standalone controller it counts sends from every
/// gateway through that controller.
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
    credits: Credits,
    ttl_micros: Option<u64>,
    _profile: String,
}

impl SentCredit {
    pub(crate) fn new(profile: &str, ttl_ms: Option<u32>) -> Self {
        Self {
            credits: Credits::default(),
            ttl_micros: ttl_ms.map(|ms| u64::from(ms) * 1000),
            _profile: profile.to_owned(),
        }
    }

    pub(crate) fn dispatched(
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
            0,
            |since, at_micros, tokens| since.record(at_micros, tokens),
        );
    }

    pub(crate) fn membership(&self, change: &Membership<'_>) {
        self.credits.membership(change, |_| {});
    }

    /// One row's sends.
    pub fn read(&self, facts: &RequestFacts<'_>, row: &Candidate) -> Sent {
        self.reads(facts).of(row)
    }

    /// One pass's reads. The pass loads the tally map once and reads each
    /// row's count when asked. A request clock of zero applies no time limit.
    pub fn reads<'a>(&'a self, facts: &RequestFacts<'_>) -> SentReads<'a> {
        SentReads {
            tally: self.credits.tally.load(),
            cutoff: self
                .ttl_micros
                .map_or(0, |ttl| facts.now_micros.saturating_sub(ttl)),
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
    /// The sends since the later of the row's sample and the time cutoff.
    /// A row without a report has a sample instant of zero.
    pub fn of(&self, row: &Candidate) -> Sent {
        self.tally
            .get(&(row.id, row.dp_rank))
            .map_or_else(Sent::default, |since| {
                let sample = row.load.map_or(0, |_| row.load_sampled_at_micros);
                let (tokens, requests) = since.credit_through(sample.max(self.cutoff));
                Sent {
                    requests: requests as u32,
                    tokens,
                }
            })
    }
}

#[cfg(test)]
mod tests;
