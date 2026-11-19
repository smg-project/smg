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
//!   [`LoadRow`], the [`AffinityFacts`] column beside the rows, [`Constraints`]
//!   and [`PoolFacts`]. Built once per decision by the host into a reused
//!   buffer; no handle, no callback into the host.
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
//!   A scorer or picker allocates nothing per decision. The consistent-hash
//!   scorer owns its ring ([`ring`]), fed by the membership hooks.
//! - 4.5, the stages in code ([`catalog`]): the [`Catalog`] of the stages a
//!   build knows by name, the stock build's and the ones a host registers.
//!   A document names either kind the same way, and the validator resolves
//!   every name once ([`PolicySpec::validate_in`]).
//! - 4.6, the schema ([`spec`]): [`PolicySpec`] (`apiVersion: smg.policy/v1`,
//!   no `kind`), its validator and the [`Needs`] summary the host reads at
//!   apply, before any request exists.
//!
//! - 5.1 and appendix A, the store ingress ([`store`]): the traits through
//!   which the controller's store is fed (membership, health, loads, KV
//!   events, reservations, sessions, leases), the one-process
//!   [`store::MemoryStore`] behind them and the contract tests over it. The
//!   KV events feed the index the store holds for each model
//!   ([`store::index`]), which no decision reads yet. A
//!   gateway in the embedded mode implements the same traits over its own
//!   structures. The store publishes what a decision reads
//!   ([`store::World`]: every pool as whole candidate rows) and builds one
//!   decision's view from it through no lock; the service that calls the
//!   build is a later step.
//! - 10, the election ([`elector`]): one [`elector::Elector`] over a
//!   [`elector::LeaseBackend`] (the lease file and the in-process record
//!   here, the Kubernetes `Lease` next), terms that only grow, and the
//!   [`elector::Fence`] a shard thread reads before every decision it sends.
//! - 5.4, replication ([`replication`]): the log of the store's replicated
//!   writes (membership and status, load rows, a term's start), each at its
//!   position under its term, told by the store through
//!   [`store::Journal`], the watched kinds kept apart from the load
//!   rows; the rejoin rule and the bounded read a standby or a watcher of
//!   the rows is served by, and the store's cut at one position
//!   ([`store::MemoryStore::cut`]). The stream, the standby and the watch
//!   are later steps.
//!
//! - Section 9, the WebAssembly plugin host (the `wasm` module, behind the `wasm`
//!   feature): user policies as components of the committed `smg:policy`
//!   package, one kept instance per decider, every call under fuel,
//!   bound to a profile as the framework's own stages. What every plugin
//!   kind shares (the breaker, the counters, the stage adapters) is the
//!   `plugins` module.
//! - Behind the `ingest` feature, the ingestion drivers (`ingest`): the load
//!   path (the `GetLoads` poll per engine under the store's poll
//!   arbitration, the metrics scrape of an engine without a servicer) and
//!   the load helpers (the row conversion, the scrape parser). The KV
//!   subscription, the prober and discovery arrive in the step's later PRs.
//!
//! - Behind the `controller` feature, the standalone controller
//!   (`controller`): the listeners with their credentialed scopes, the
//!   health service on each, the admin endpoints, readiness and the drain,
//!   the load driver over the store, and the binary of the package's name
//!   over them. The scopes' calls and the election arrive in the step's
//!   later PRs.
//! - Behind the `client` feature, the reference client of the controller's
//!   protocol (`client`): one gateway's side of it (the dial, `Leader`,
//!   the session's `Hello` answered by a `Welcome`, and a request's
//!   lifecycle reports on that session), with a blocking form. No shipped
//!   binary builds it.
//!
//! What this step does not hold: the gRPC plugin service, the operator calls,
//! the other ingestion drivers and the gRPC API; they arrive in later
//! steps and build on these types. The
//! in-process view carries two host instants (the decision's clock and a
//! report's sample instant); the wire carries durations only.

pub mod affinity;
pub mod bucket;
pub mod builtins;
pub mod catalog;
#[cfg(feature = "client")]
pub mod client;
#[cfg(feature = "controller")]
pub mod controller;
pub mod counts;
pub mod decider;
pub mod decision;
pub mod elector;
pub mod framework;
#[cfg(feature = "controller")]
pub mod grpc;
#[cfg(feature = "ingest")]
pub mod ingest;
pub mod lease;
#[cfg(any(feature = "wasm", feature = "controller"))]
pub mod plugins;
#[cfg(feature = "controller")]
pub mod proto;
pub mod replication;
pub mod ring;
pub mod sessions;
pub mod spec;
pub mod store;
pub mod view;
#[cfg(feature = "wasm")]
pub mod wasm;
pub mod wit;

pub use catalog::{Catalog, CatalogError, StageKind};
pub use counts::{KeptCounts, LentCounts};
pub use decision::{
    Decision, Lifted, Pair, Pick, Queue, Shed, ShedReason, Survivors, Target, Targets, Trace,
    MAX_TARGETS, TRACE_TOP,
};
pub use framework::{
    Binding, DecisionScope, EncodeItems, EncodePlaced, EncodePool, EncoderFacts, Engine, Filter,
    Front, Narrowed, PairPools, PairTable, PairTableError, Picker, Plan, Profile, Scorer, Scratch,
    Spread, TieBand, Unit, Weighted, DRAW_BOUND, DRAW_FROM, RANK_FROM,
};
pub use lease::{
    EventKind, EventSet, Lease, LeaseId, LeaseKind, Lifecycle, Membership, Outcome, ReleaseReason,
    Side, Signal, TransportClass,
};
pub use ring::{ConsistentHash, Ring};
pub use sessions::{
    Assignment, Forgets, Pins, Recovery, SessionRule, SessionTable, SessionVerdict,
};
pub use spec::{
    AppliedSpec, Needs, PolicySpec, PowerOfTwoMetric, SessionsSpec, SpecError, API_VERSION, PRESETS,
};
pub use view::{
    fingerprint, AffinityFacts, Candidate, Constraints, Engines, HashKind, Header, Health,
    Identity, LoadRow, MediaCounts, ModelFingerprint, NamedAffinity, NamedHolder, OverloadRegime,
    Phase, PinSource, PoolFacts, RequestFacts, Role, Session, SignalSource, WireModes, WorkerId,
    ZoneCapacity,
};

/// SplitMix64 over a decision's seed: the only randomness a stage gets, so a
/// decision replays from its trace.
///
/// The streams a decision opens are listed here, so that a new site derives
/// one that none of them reads:
///
/// - the seed's own stream: the profile's picker, and a plugin's draws. A
///   picker that draws first reads it too, and whatever follows that draw
///   (the order of the rows drawn, or the whole pass behind a draw that kept
///   nobody) reads it past those values;
/// - one value of the seed's stream, read behind the draw of a picker whose
///   unit is the engine: the seed of the stream that draws between equal
///   rows of the engine taken;
/// - the seed under the `warmup-slice` scorer's mask: that scorer's draws;
/// - the seed under the alternates' mask: the order of equal counts among
///   the workers the count-first picker passed over;
/// - the first value of the seed's stream: the seed a pair plan hands its
///   decode side, whose stages derive their streams from it as above;
/// - the seed under the store's two naming masks: where the store names a
///   decision's rows and cuts a list at the profile's sample, the holders'
///   draw and the warming rows' draw;
/// - the first value of the holders' stream, and of each stream so seeded:
///   the seed of each lookup a naming makes again, a stream a round.
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

    /// The stream's state: [`Draw::new`] of it continues the stream past
    /// the values read.
    pub(crate) const fn state(self) -> u64 {
        self.0
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
