//! A binding's shadow profile next to a live decision: its outcome, the
//! part of a request's budget that must be left for it to run, and its
//! counts.

#![cfg_attr(
    not(feature = "in-process"),
    expect(dead_code, reason = "Only the engine over worker lists asks a shadow.")
)]

use std::{sync::atomic::AtomicU64, time::Duration};

use super::super::{
    decision::{PairSide, ShadowKind},
    view::{RequestFacts, WorkerId},
};

/// How many of a generation's bindings count their shadows in a row of
/// their own ([`ShadowCount::binding`]). The bindings behind them count in
/// the last row.
pub const SHADOW_ROWS: usize = 16;

/// One decider's counts of its shadows, on lines of their own: a row a
/// binding by its position in its generation, a count a kind. A table
/// for each place a shadow is asked ([`ShadowSide`]).
#[derive(Debug, Default)]
#[repr(align(128))]
pub(crate) struct ShadowCounts([[[AtomicU64; ShadowKind::COUNT]; SHADOW_ROWS]; ShadowSide::COUNT]);

impl ShadowCounts {
    pub(crate) fn of(&self, binding: u32, side: ShadowSide, kind: ShadowKind) -> &AtomicU64 {
        let row = (binding as usize).min(SHADOW_ROWS - 1);
        &self.0[side.index()][row][kind.index()]
    }
}

/// Where a shadow is asked: the table it is counted in, and the side a
/// host shows its counts by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ShadowSide {
    /// Beside a single request's decision.
    Single,
    /// Beside the prefill side of a pair, at the pair's first decision.
    Prefill,
    /// Beside the decode side of a pair, at the pair's first decision.
    Decode,
    /// Beside a decode decided after its prefill, at that second
    /// decision. A first decision that asked its decode side counted it
    /// beside the decode it named first: the two are never one count.
    DecodeAgain,
}

impl ShadowSide {
    /// How many sides there are: the tables of a decider's counts, and
    /// the width of a row's word of which sides it counts by.
    pub(crate) const COUNT: usize = 4;

    /// Every side, in the order of its place: the order a host is told a
    /// binding's counts.
    pub(crate) const ALL: [Self; Self::COUNT] =
        [Self::Single, Self::Prefill, Self::Decode, Self::DecodeAgain];

    /// The side's place among the sides, under [`Self::COUNT`]. Every side
    /// is named here: one added to the enum does not build until it takes
    /// a place.
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Single => 0,
            Self::Prefill => 1,
            Self::Decode => 2,
            Self::DecodeAgain => 3,
        }
    }

    /// Where the shadow of `side` of a pair is asked at the pair's first
    /// decision.
    pub(crate) const fn of_pair(side: PairSide) -> Self {
        match side {
            PairSide::Prefill => Self::Prefill,
            PairSide::Decode => Self::Decode,
        }
    }

    /// The side's label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Single => "single",
            Self::Prefill => "prefill",
            Self::Decode => "decode",
            Self::DecodeAgain => "decode-again",
        }
    }
}

/// One count of an engine's shadow decisions, as a host is told it
/// (`InProcess::shadow_counts`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct ShadowCount {
    /// The row of the binding the shadow is of: the binding's position in
    /// its document, from 0. Every binding from the last row on is told as
    /// that row, [`SHADOW_ROWS`] less one.
    pub binding: u32,
    /// Where the shadow was asked.
    pub side: ShadowSide,
    /// How the shadow's decision stood beside the live one.
    pub kind: ShadowKind,
    /// How many shadow decisions the engine counted so, under its document.
    pub count: u64,
}

/// What a request's budget must still hold at the end of its live
/// decision for its shadow to be asked. A shadow asked with less could
/// take a served answer past its budget, where the decider's last read
/// refuses it.
pub(crate) const SHADOW_FLOOR: Duration = Duration::from_millis(1);

/// What a request's own budget holds, as its host states it, for the
/// floor a shadow is held to ([`SHADOW_FLOOR`]): a request that states
/// none has no floor to hold.
pub(crate) fn budget_left(facts: &RequestFacts<'_>) -> Duration {
    match facts.budget_micros {
        0 => Duration::MAX,
        micros => Duration::from_micros(micros.into()),
    }
}

/// What a decision's shadow said beside it, where one was due.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Shadowed {
    /// The position of the request's binding in its generation.
    pub(crate) binding: u32,
    /// The shadow's decision beside the live one, and
    /// [`ShadowKind::Skipped`] where the shadow was due and not asked.
    pub(crate) kind: ShadowKind,
    /// The first target of the shadow's pick.
    pub(crate) first: Option<WorkerId>,
}

/// What became of a decision's shadow.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Shadow {
    /// None was due: the request's binding asks none, or the door made
    /// the decision itself over a view its gateway left no row.
    NotDue,
    /// One was due: what it said, or that it was not asked.
    Said(Shadowed),
    /// The shadow panicked, and the panic was held where it was asked.
    Faulted,
}
