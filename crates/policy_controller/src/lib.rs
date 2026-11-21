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
//!   buffer; no handle, no callback into the host. The values a report
//!   states under a name the document declares stand beside the rows too
//!   ([`load`]): a stage reads one by its [`SignalColumn`].
//! - 4.2, the decision ([`engine`]): an ordered [`Pick`] of up to
//!   [`MAX_TARGETS`] targets, a [`Shed`] with its reason and the stage that
//!   emptied the pool, or a [`Queue`] ticket (specified in v1; no built-in
//!   issues one yet). A decision books nothing by itself: its door files
//!   the pick's lease at the issue, which books the first target, and the
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
//!   prefixes reads that record at the naming through the in-process door.
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
//!   ([`store::index`]), which no decision reads yet. A
//!   gateway in the embedded mode implements the same traits over its own
//!   structures. The store publishes what a decision reads
//!   ([`store::StoreSnapshot`]: every pool as whole candidate rows) and builds
//!   one decision's view from it through no lock; the service that calls the
//!   build is a later step.
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
//!   ([`store::MemoryStore::cut`]). The stream, the standby and the watch
//!   are later steps.
//!
//! - Section 9, the WebAssembly plugin host (the `plugins::wasm` module, behind the `wasm`
//!   feature): user policies as components of the committed `smg:policy`
//!   package, one kept instance per decider, every call under fuel,
//!   bound to a profile as the framework's own stages. What every plugin
//!   kind shares (the breaker, the counters, the stage adapters) is the
//!   `plugins::shared` module.
//! - Behind the `ingest` feature, the ingestion drivers (`ingest`): the load
//!   path (the `GetLoads` poll per engine under the store's poll
//!   arbitration, the metrics scrape of an engine without a servicer) and
//!   the metrics parser. The load driver and the worker host's load adapter
//!   share the `load_report` conversion.
//!
//! - Behind the `operator` feature, `document::operator::DocumentManager`
//!   applies a host's start document and serves its status, document and dry
//!   run. The feature also serves worker snapshots over the host's store.
//! - The door ([`engine`]): the order of one decision's calls, written once
//!   for every host. A host that embeds the engine asks
//!   [`engine::StoreBackedEngine`], which runs the order on the caller's thread
//!   over a [`store::MemoryStore`]: the lease is filed at the issue, the host
//!   says what became of it ([`engine::StoreBackedEngine::report`]), itself or
//!   through a handle, and its timer calls the door's clock
//!   ([`engine::StoreBackedEngine::sweep`]). The standalone controller runs the
//!   same order on its deciders' threads.
//! - The handle ([`tracking`]): one decision held until its request ends.
//!   A host that holds a [`tracking::DecisionGuard`] says the target it took,
//!   the dispatch, and the completion or the release. The handle reports to the
//!   in-process door or, with the `client` feature, over a link.
//! - Behind the `controller` feature, the standalone controller
//!   (`controller`): the listeners with their credentialed scopes, the
//!   health service on each, the admin endpoints, readiness, the drain and
//!   the load driver over the store. The stock binary also requires the
//!   `worker-host` feature so it can read cache events.
//! - Behind the `client` feature, the gateway's side of the controller's
//!   protocol (`client`). The reference client (`service::client::Client`) asks
//!   one request at a time, reports its lifecycle and watches workers'
//!   rows, with a blocking form. A host's requests share `service::client::Link`
//!   for concurrent asks on one decide stream and reports on one session.
//! - Behind the same feature, the watcher (`watcher`): a gateway's copy of
//!   the controller's workers. It takes the watch's events in their order
//!   into a [`store::MemoryStore`] of its own, through the door a
//!   standby's applier writes a followed row by, and names the place a
//!   watch opened again goes on from.
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
//! - Behind the `discovery` feature, worker discovery under the controller:
//!   the binary takes the discovery flags, and the replica that leads runs
//!   the selected provider over the replica's worker objects. A pass asks
//!   the job queue the host was handed, and defers what it found while none
//!   stands.
//!
//! The in-process view carries two host instants (the decision's clock and a
//! report's sample instant); the wire carries durations only.
//!
//! Import paths follow two rules. The folders `engine`, `document`, `load`,
//! `tracking` and `stages` list their public items themselves, so an item is
//! at its folder: `engine::WorkerId`. A folder of named things keeps the name
//! in the path: a policy (`policies::least_load::LeastLoad`), a plugin kind
//! (`plugins::wasm::Plugin`). `store`, `worker_host`, `service` and `plugins`
//! are not under one path yet: some of their items have a second path,
//! through their file, such as `plugins::wasm::config::PluginSpec`.
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

#[cfg(any(feature = "ingest", feature = "operator"))]
mod address;
pub mod document;
pub mod engine;
pub mod load;
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
///   the seed of each lookup a naming makes again, a stream a round;
/// - the first value of the seed under the store's mask for refused
///   holders, and of each stream so seeded: the seed of each lookup a
///   naming makes again past a cap of holders the pass refused, a stream a
///   round.
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
