//! The routing policy engine.
//!
//! A policy sees a bounded, handle-free [`view`] of one request and its
//! candidates, answers a typed [`decision`] (an ordered list of targets, or a
//! shed), and learns outcomes through the [`lease`] lifecycle. Policies are
//! composed from stages in the [`framework`]: a profile runs filters, weighted
//! scorers and one picker; a plan sequences profiles for a request (a single
//! pick, or a prefill pick followed by a decode pick over the chosen prefill's
//! partners). The [`spec`] is the one versioned document that names profiles,
//! plans and bindings and tells the host what the policies need before any
//! request exists.
//!
//! The crate knows nothing about the host: no worker handles, no registry, no
//! transport. A host builds the view, calls [`framework::Engine::decide`],
//! commits one target under its own locks and reports the lifecycle back.

pub mod builtins;
pub mod decision;
pub mod framework;
pub mod lease;
pub mod spec;
pub mod view;

pub use decision::{Decision, Pick, Shed, ShedReason, Target, Targets, Trace, MAX_TARGETS};
pub use framework::{Engine, Filter, PairPools, Picker, Profile, Scorer, Scratch, Weighted};
pub use lease::{Lease, LeaseId, LeaseKind, Lifecycle, Membership, Outcome, Side, Signal};
pub use spec::{Needs, PolicySpec, SpecError, API_VERSION};
pub use view::{
    Candidate, Constraints, HashKind, Health, LoadReport, MediaCounts, OverloadRegime, Phase,
    PoolFacts, RequestFacts, Role, SignalSource, WorkerId,
};

/// SplitMix64 over a decision's seed: the only randomness a stage gets, so a
/// decision replays from its trace.
#[derive(Clone, Copy, Debug)]
pub struct Draw(u64);

impl Draw {
    /// A draw sequence from a decision's seed.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// The next 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A uniform index below `n` (`0` when `n` is `0`).
    pub fn below(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        // Rejection-free: the bias of a modulo over 64 bits is below 2^-32
        // for any pool a gateway holds.
        (self.next_u64() % n as u64) as usize
    }
}

#[cfg(test)]
mod tests {
    use super::Draw;

    #[test]
    fn the_draw_is_deterministic_per_seed_and_stays_below_the_bound() {
        let mut a = Draw::new(7);
        let mut b = Draw::new(7);
        for _ in 0..64 {
            let x = a.below(13);
            assert_eq!(x, b.below(13));
            assert!(x < 13);
        }
        assert_eq!(Draw::new(1).below(0), 0);
    }
}
