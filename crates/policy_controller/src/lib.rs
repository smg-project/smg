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
//! - 4.1, the view ([`engine`]): [`RequestFacts`], [`Candidate`] rows with their
//!   [`LoadRow`], the [`AffinityFacts`] column beside the rows, [`Constraints`]
//!   and [`PoolFacts`]. Built once per decision by the host into a reused
//!   buffer; no handle and no callback into the host, but for the hash
//!   ring a host may lend with a request. The values a report
//!   states under a name the document declares stand beside the rows too
//!   ([`load`]): a stage reads one by its [`SignalColumn`].
//! - 4.2, the decision ([`engine`]): an ordered [`Pick`] of up to
//!   [`MAX_TARGETS`] targets, a [`Shed`] with its reason and the stage that
//!   emptied the pool, or a [`Queue`] ticket (specified in v1; no built-in
//!   issues one yet). A decision books nothing by itself: the pick's
//!   lease is filed at the issue, which books the first target, and the
//!   host reports the lifecycle back. One issue books another target: a
//!   pair's decode asked for later and sent with no decode call is issued
//!   at its dispatch, on the target that dispatch took.
//! - 4.3, the lifecycle ([`tracking`]): the [`Lease`] a decision carries and the
//!   [`Lifecycle`] hooks a stateful stage learns from (issued, dispatched,
//!   completed, released, membership, signal).
//! - 4.4, the framework ([`engine`]): a [`Profile`] runs [`Filter`]s,
//!   weighted [`Scorer`]s and one [`Picker`]; a [`Plan`] sequences profiles for
//!   a request; the [`Engine`] resolves a request to its binding and decides.
//!   A scorer or picker allocates nothing per decision. The consistent-hash
//!   scorer owns its ring ([`policies::consistent_hashing`]), fed by the
//!   membership hooks. The
//!   prefixes sent to a model's workers without cache events are kept by
//!   the sent-prefix predictor ([`tracking`]). A profile that reads sent
//!   prefixes reads that record at the naming.
//! - 4.5, the stages in code ([`stages`]): the [`StageRegistry`] of the stages
//!   a build knows by name, the stock build's and the ones a host registers. A
//!   document names either kind the same way, and the validator resolves every
//!   name once ([`PolicySpec::validate_in`]).
//! - 4.6, the schema ([`document`]): [`PolicySpec`] (`apiVersion: smg.policy/v1`,
//!   no `kind`), its validator and the [`Needs`] summary the host reads at
//!   apply, before any request exists. A binding names where a request's
//!   routing key comes from, and one function derives it ([`document`]).
//!   A gateway's routing settings mean a document too, and
//!   [`document::RoutingSettings`] writes it.
//!
//! - 5.1 and appendix A, the store ingress ([`store`]): the traits through
//!   which the controller's store is fed (membership, health, loads, KV
//!   events, reservations, sessions, leases), the one-process
//!   [`store::MemoryStore`] behind them and the contract tests over it. The
//!   KV events feed the index the store holds for each model
//!   ([`store`]), which no decision reads yet. A
//!   gateway in the embedded mode implements the same traits over its own
//!   structures.
//! - 10, the election ([`service::elector`]): one [`service::elector::Elector`] over a
//!   [`service::elector::LeaseBackend`] (the lease file and the in-process record
//!   here, the Kubernetes `Lease` next), terms that only grow, and the
//!   [`service::elector::Fence`] a shard thread reads before every decision it sends.
//! - 5.4, replication ([`service::replication`]): the log of the store's replicated
//!   writes (membership and status, load rows, a term's start), each at its
//!   position under its term, told by the store through
//!   [`store::Journal`], the watched kinds kept apart from the load
//!   rows; the rejoin rule and the bounded read a standby or a watcher of
//!   the rows is served by, and the store's cut at one position
//!   ([`store::MemoryStore::checkpoint`]). The stream, the standby and the
//!   watch are later steps.
//!
//! - Section 9, the WebAssembly plugin host (the `plugins::wasm` module, behind the `wasm`
//!   feature): user policies as components of the committed `smg:policy`
//!   package, one kept instance per decider, every call under fuel,
//!   bound to a profile as the framework's own stages. What every plugin
//!   kind shares (the breaker, the counters, the stage adapters) is the
//!   `plugins::shared` module.
//! - Behind the `operator` feature, `document::operator::DocumentManager`
//!   applies a host's start document and serves its status, document and dry
//!   run. The feature also serves worker snapshots over the host's store.
//! - The engine ([`engine`]): an embedded host lends its worker lists to
//!   `engine::PolicyEngine` and asks it on the caller's thread. The
//!   standalone controller asks the same engine on its deciders' threads.
//! - The handle ([`tracking`]): one decision held until its request ends.
//!   A host that holds a [`tracking::DecisionGuard`] says the target it took,
//!   the dispatch, and the completion or the release. The handle reports to the
//!   in-process door or, with the `client` feature, over a link.
//! - Behind the `controller` feature, the standalone controller
//!   (`controller`): the listeners with their credentialed scopes, the
//!   health service on each, the admin endpoints, readiness, the drain and
//!   the load monitor's task. The feature turns on `in-process`, the
//!   engine over worker objects, and `worker-host`, so every controller
//!   keeps its worker objects, probes them and reads cache events.
//! - Behind the `client` feature, the gateway's side of the controller's
//!   protocol (`client`). The reference client (`service::client::Client`) asks
//!   one request at a time, reports its lifecycle and watches workers'
//!   rows, with a blocking form. A host's requests share `service::client::Link`
//!   for concurrent asks on one decide stream and reports on one session.
//! - Behind the same feature, the watcher (`watcher`): a gateway's copy of
//!   the controller's workers. It takes the watch's events in their order
//!   into a [`store::MemoryStore`] of its own, through the holder of the
//!   gateway's worker objects as a standby's applier writes a followed
//!   row, and names the place a watch opened again goes on from.
//! - Behind the same feature, the worker follower
//!   (`service::worker_follower::WorkerFollower`): the task a gateway
//!   follows a controller's worker list with. It dials the controller,
//!   keeps the watcher's watch open, waits and dials again after an error,
//!   and goes to the leader a replica names.
//! - Behind the `worker-host` feature, a replica's worker objects
//!   (`worker_host`): one object of the worker crate for every row of the
//!   membership store, in a registry the host hands in, built, replaced and
//!   dropped as the rows come, change and go. A controller built with it
//!   follows its own store, on a leader and on a standby alike, and hands
//!   each object a gRPC connector: an object connects to its engine at its
//!   first use.
//! - Behind the `in-process` feature, the engine in a gateway's own process
//!   (`engine::PolicyEngine`): it gives the gateway's own placement its pick
//!   over the gateway's worker objects, read in place. It keeps no worker
//!   list, no health and no count of requests in flight, and names nothing
//!   of the store.
//! - Behind the `discovery` feature, worker discovery under the controller:
//!   the binary takes the discovery flags, and the replica that leads runs
//!   the selected provider over the replica's worker objects. A pass asks
//!   the job queue the host was handed, and defers what it found while none
//!   stands.
//!
//! The in-process view carries two host instants (the decision's clock and a
//! report's sample instant); the wire carries durations only.
//!
//! Every public item has one import path. A folder lists the public items of
//! its files, so an item is at its folder: `engine::WorkerId`,
//! `store::MemoryStore`. A module that is a named thing keeps its name in the
//! path: a policy (`policies::least_load::LeastLoad`), a plugin kind
//! (`plugins::wasm::Plugin`), a part of the service
//! (`service::controller::Config`). So does a file whose names read only
//! with its own: `store::pairing::key`, `service::controller::metrics::STATE`.
//!
//! [`RequestFacts`]: engine::RequestFacts
//! [`Candidate`]: engine::Candidate
//! [`LoadRow`]: engine::LoadRow
//! [`AffinityFacts`]: engine::AffinityFacts
//! [`Constraints`]: engine::Constraints
//! [`PoolFacts`]: engine::PoolFacts
//! [`SignalColumn`]: load::SignalColumn
//! [`Pick`]: engine::Pick
//! [`MAX_TARGETS`]: engine::MAX_TARGETS
//! [`Shed`]: engine::Shed
//! [`Queue`]: engine::Queue
//! [`Lease`]: tracking::Lease
//! [`Lifecycle`]: stages::Lifecycle
//! [`Profile`]: engine::Profile
//! [`Filter`]: engine::Filter
//! [`Scorer`]: engine::Scorer
//! [`Picker`]: engine::Picker
//! [`Plan`]: engine::Plan
//! [`Engine`]: engine::Engine
//! [`StageRegistry`]: stages::StageRegistry
//! [`PolicySpec::validate_in`]: document::PolicySpec::validate_in
//! [`PolicySpec`]: document::PolicySpec
//! [`Needs`]: document::Needs
//!

use std::{cell::Cell, collections::hash_map::RandomState, hash::BuildHasher, thread};

#[cfg(feature = "operator")]
mod address;
pub mod document;
pub mod engine;
pub mod load;
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub mod mirrored;
#[cfg(any(feature = "wasm", feature = "controller", feature = "in-process"))]
mod names;
pub mod plugins;
pub mod policies;
pub mod service;
pub mod stages;
pub mod store;
pub mod tracking;
#[cfg(feature = "worker-host")]
pub mod worker_host;

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
/// - the seed under a mask made of a stage's name ([`Draw::for_stage`]): the
///   draws of a filter or a scorer that opens it, the call for one its host
///   registers;
/// - one value of the seed's stream, read behind the draw of a picker whose
///   unit is the engine: the seed of the stream that draws between equal
///   rows of the engine taken;
/// - the seed under the `warmup-slice` scorer's mask: that scorer's draws;
/// - the seed under the alternates' mask: the order of equal counts among
///   the workers the count-first picker passed over;
/// - the first value of the seed's stream: the seed a pair plan hands its
///   decode side, whose stages derive their streams from it as above;
/// - the seed under the two naming masks: where a decision's rows are named
///   and a list is cut at the profile's sample, the holders' draw and the
///   warming rows' draw.
#[derive(Clone, Copy, Debug)]
pub struct Draw(u64);

impl Draw {
    /// A draw sequence from a decision's seed.
    pub const fn new(seed: u64) -> Self {
        Self(seed)
    }

    /// A stage's own stream of the seed it is handed: the seed under a mask
    /// made of `stage`, the name the stage answers by. It is the call for a
    /// filter or a scorer its host registers, where that stage draws:
    /// `Draw::for_stage(facts.seed, self.name())`. The stream of
    /// [`Draw::new`] of the seed is the one the profile's picker reads: a
    /// stage that drew from it would read the picker's values, one for one,
    /// and where the picker draws, the worker it takes would follow what
    /// the stage drew.
    ///
    /// The mask is the fingerprint of the name's bytes
    /// ([`engine::fingerprint`]), which is never zero, so the stream is
    /// never the seed's own. Two names of two fingerprints open two
    /// streams, and a catalog of stages holds a name once
    /// ([`stages::StageRegistry`]). The stream is another of those listed
    /// on [`Draw`] only where two 64-bit values meet.
    ///
    /// The stream is that of
    /// `Draw::new(seed ^ fingerprint(stage.as_bytes()))`: a stage may keep
    /// its name's fingerprint from its build and open its stream with one
    /// exclusive-or at each call. Every call opens the stream at its start:
    /// under one seed and one name it draws the same values every time.
    #[must_use]
    pub fn for_stage(seed: u64, stage: &str) -> Self {
        Self(seed ^ engine::fingerprint(stage.as_bytes()))
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

/// A seed no other request of this thread was given: the one randomness a
/// stage gets ([`engine::RequestFacts::seed`]), and what a picker
/// that places by itself draws from. Under a constant seed a policy that
/// draws picks the same worker of an unchanged pool every time.
///
/// Each thread walks a draw sequence of its own ([`Draw`]). Its start is
/// the standard library's process-random hash of the thread's id, so two
/// threads and two runs of a process start apart.
pub(crate) fn fresh_seed() -> u64 {
    thread_local! {
        static NEXT: Cell<u64> = Cell::new(RandomState::new().hash_one(thread::current().id()));
    }
    NEXT.with(|next| {
        let mut draw = Draw::new(next.get());
        let seed = draw.next_u64();
        next.set(draw.state());
        seed
    })
}

#[cfg(test)]
mod tests {
    use super::Draw;
    use crate::engine::fingerprint;

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

    /// A stage's stream shares no value, place for place, with the seed's
    /// own stream, which the picker reads: under the empty name too. It is
    /// the seed's stream under the fingerprint of the stage's name: the
    /// equality a stage that keeps its fingerprint opens its stream by.
    #[test]
    fn a_stages_stream_is_the_seeds_under_its_names_fingerprint() {
        for seed in [0, 1, 7, 0x9e37_79b9_7f4a_7c15, u64::MAX] {
            for name in ["pack", "spread", "a", ""] {
                let mut own = Draw::for_stage(seed, name);
                let mut pickers = Draw::new(seed);
                let mut by_hand = Draw::new(seed ^ fingerprint(name.as_bytes()));
                for place in 0..8 {
                    let value = own.next_u64();
                    assert_ne!(value, pickers.next_u64(), "{name} at {seed}, {place}");
                    assert_eq!(value, by_hand.next_u64(), "{name} at {seed}, {place}");
                }
            }
        }
    }

    /// The two names, of two fingerprints, open two streams from one seed.
    /// One name opens the same stream from one seed every time: the call
    /// keeps nothing between two calls.
    #[test]
    fn two_names_open_two_streams_and_one_name_replays() {
        for seed in 0..256u64 {
            let first = |name| Draw::for_stage(seed, name).next_u64();
            assert_ne!(first("pack"), first("spread"), "seed {seed}");
            assert_eq!(first("pack"), first("pack"), "seed {seed}");
        }
    }
}
