//! A binding's shadow profile next to a live decision: its outcome, the
//! part of a request's budget that must be left for it to run, and its
//! counts.

#![cfg_attr(
    not(feature = "in-process"),
    expect(dead_code, reason = "Only the engine over worker lists asks a shadow.")
)]

use std::{sync::atomic::AtomicU64, time::Duration};

use super::{decision::ShadowKind, framework::PairSide, view::WorkerId};

/// How many of a generation's bindings count their shadows in a row of
/// their own ([`DecisionEngine::shadowed`]). The bindings behind them count
/// in the last row. A host that shows the counts walks this many rows.
///
/// [`DecisionEngine::shadowed`]: super::decision_engine::DecisionEngine::shadowed
pub const SHADOW_ROWS: usize = 16;

/// One decider's counts of its shadows, on lines of their own: a row a
/// binding by its position in its generation, a count a kind. A table
/// for each place a shadow is asked ([`Counted`]).
#[derive(Debug, Default)]
#[repr(align(128))]
pub(crate) struct ShadowCounts([[[AtomicU64; ShadowKind::COUNT]; SHADOW_ROWS]; 4]);

impl ShadowCounts {
    pub(crate) fn of(&self, binding: u32, table: Counted, kind: ShadowKind) -> &AtomicU64 {
        let table = match table {
            Counted::Single => 0,
            Counted::Side(side) => 1 + side.index(),
            Counted::Later => 3,
        };
        let row = (binding as usize).min(SHADOW_ROWS - 1);
        &self.0[table][row][kind.index()]
    }
}

/// Where a shadow was asked: the table it is counted in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Counted {
    /// Beside a single's decision.
    Single,
    /// Beside this side of a pair, at the pair's decision.
    #[expect(
        dead_code,
        reason = "no host asks a pair's shadows until the engine over worker lists does"
    )]
    Side(PairSide),
    /// Beside a decode decided after its prefill, at that second
    /// decision. The pair's first decision counted its decode side
    /// beside the decode it named first: the two are never one count.
    #[expect(
        dead_code,
        reason = "no host asks a later decode's shadow until the engine over worker lists does"
    )]
    Later,
}

/// What a request's budget must still hold at the end of its live
/// decision for its shadow to be asked. A shadow asked with less could
/// take a served answer past its budget, where the decider's last read
/// refuses it.
pub(crate) const SHADOW_FLOOR: Duration = Duration::from_millis(1);

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
