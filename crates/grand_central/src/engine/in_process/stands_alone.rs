//! A request of a side's phase that is no side of a pair.
//!
//! A document binds a model and a phase to a plan. A pair plan decides some
//! phases together, at the pair's prefill ([`Plan::pairs`]). A request of
//! any other phase stands alone, whatever its phase says: one profile
//! decides it over the one list its host hands, by the binding its own
//! phase resolves to.
//!
//! So a host that hands ONE list asks for one worker of it, and the engine
//! answers where the document agrees. A prefill under a plan of one profile
//! stands alone. A prefill under a pair plan is its pair's first side: over
//! one list it has no decode to pair with, and it is decided over no
//! worker. A decode or an encode request that is a pair's side has no
//! decision of its own.
//!
//! [`Plan::pairs`]: crate::engine::framework::Plan::pairs

use super::Loaded;
use crate::engine::view::Phase;

impl Loaded<'_, '_> {
    /// Whether a made request of `model` at `phase` stands alone: a single
    /// request, and a request of a phase that its binding's plan decides
    /// with no pair. One profile decides it over the one list its host
    /// hands. A request that no binding matches stands alone too: over a
    /// list the whole decision declines it by name, and over none it is
    /// decided over no worker, as a single request is.
    #[inline]
    pub(super) fn stands_alone(self, model: &str, phase: Phase) -> bool {
        phase == Phase::Single || !self.pairs(model, phase)
    }

    /// Whether the plan `model` is bound to for `phase`, by that phase's
    /// own lookup, decides the phase as a side of a pair. Out of line: a
    /// single request never asks.
    #[cold]
    fn pairs(self, model: &str, phase: Phase) -> bool {
        self.document.spec.engine().pairs(model, phase)
    }
}

#[cfg(test)]
#[path = "../../../tests/unit/engine/in_process/stands_alone/tests.rs"]
mod tests;
