//! The policy controller: the engine (contract, framework, built-ins, plugin
//! hosts, schema, operator calls; no gateway dependency) and, in later steps
//! behind a controller feature, the store, ingestion, gRPC API and election.
//!
//! This step carries the engine of the policy-controller design (revision 3):
//! one bounded view, one typed decision, one lifecycle, and a filter, score and
//! pick framework under one versioned spec. The same code decides inside a
//! gateway process or in the standalone controller this package's binary
//! becomes, and its types are the in-process form of the design's wire
//! contracts, field for field, so a plugin host needs no second definition.
//! Section by section:
//!
//! - 4.1, the view ([`view`]): [`RequestFacts`], [`Candidate`] rows with their
//!   [`LoadRow`], [`Constraints`] and [`PoolFacts`]. Built once per decision by
//!   the host into a reused buffer; no handle, no callback into the host.
//! - 4.2, the decision ([`decision`]): an ordered [`Pick`] of up to
//!   [`MAX_TARGETS`] targets, a [`Shed`] with its reason and the stage that
//!   emptied the pool, or a [`Queue`] ticket (specified in v1; no built-in
//!   issues one yet). A decision books nothing: the host commits one target
//!   under its own locks and reports the lifecycle back.
//! - 4.3, the lifecycle ([`lease`]): the [`Lease`] a decision carries and the
//!   [`Lifecycle`] hooks a stateful stage learns from (issued, dispatched,
//!   completed, released, membership, signal).
//! - 4.4, the framework ([`framework`]): a [`Profile`] runs [`Filter`]s,
//!   weighted [`Scorer`]s and one [`Picker`]; a [`Plan`] sequences profiles for
//!   a request; the [`Engine`] resolves a request to its binding and decides.
//!   A scorer or picker allocates nothing per decision.
//! - 4.6, the schema ([`spec`]): [`PolicySpec`] (`apiVersion: smg.policy/v1`,
//!   no `kind`), its validator and the [`Needs`] summary the host reads at
//!   apply, before any request exists.
//!
//! - 5.1 and appendix A, the store ingress ([`store`]): the traits through
//!   which the controller's store is fed (membership, health, loads, KV event
//!   cursors, reservations, sessions, leases), the one-process
//!   [`store::MemoryStore`] behind them and the contract tests over it. A
//!   gateway in the embedded mode implements the same traits over its own
//!   structures; nothing decides through them yet.
//!
//! What this step does not hold: the plugin hosts, the operator calls, the
//! ingestion drivers, the gRPC API and the election; they arrive in later
//! steps and build on these types. The
//! in-process view carries two host instants (the decision's clock and a
//! report's sample instant); the wire carries durations only.

pub mod builtins;
pub mod decision;
pub mod framework;
pub mod lease;
pub mod spec;
pub mod store;
pub mod view;

pub use decision::{
    Decision, Pair, Pick, Queue, Shed, ShedReason, Target, Targets, Trace, MAX_TARGETS, TRACE_TOP,
};
pub use framework::{
    Binding, Engine, Filter, PairPools, Picker, Plan, Profile, Scorer, Scratch, TieBand, Weighted,
};
pub use lease::{
    Lease, LeaseId, LeaseKind, Lifecycle, Membership, Outcome, ReleaseReason, Side, Signal,
    TransportClass,
};
pub use spec::{Needs, PolicySpec, SpecError, API_VERSION, PRESETS};
pub use view::{
    fingerprint, Candidate, Constraints, HashKind, Header, Health, Identity, LoadRow, MediaCounts,
    OverloadRegime, Phase, PoolFacts, RequestFacts, Role, SignalSource, WireModes, WorkerId,
    ZoneCapacity,
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
