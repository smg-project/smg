//! A policy controller of your own: the stock controller with one scorer
//! added in code. The scorer is a type of this file; the built-in stages,
//! the command line, the listeners and the election are the crate's.
//!
//! A change to the view a stage reads ([`Candidate`], [`RequestFacts`]) is a
//! change to what this file compiles against, as it is for a plugin's world.

use std::process::ExitCode;

use smg_policy_controller::{
    controller, Candidate, Catalog, CatalogError, Lifecycle, Needs, RequestFacts, Scorer, TieBand,
};

/// The process's allocator, the one the stock controller's binary registers,
/// on the targets where the settings below hold. An Apple target is left out:
/// the allocator compiles no background thread there. The library registers
/// none: a controller built on it chooses its own. A burst of registrations
/// builds the store's world again at every write, and the system allocator
/// keeps resident what those builds free.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[global_allocator]
static ALLOCATOR: tikv_jemallocator::Jemalloc = tikv_jemallocator::Jemalloc;

/// The allocator's settings, the stock binary's: a background thread hands
/// freed pages back to the system ten seconds after their last use, the
/// threads share sixteen arenas at most, and an allocation up to 32 MiB takes
/// the room the last one freed. The allocator reads them from this symbol as
/// it starts, and the `_RJEM_MALLOC_CONF` environment variable after it,
/// entry by entry.
#[cfg(all(
    not(target_env = "msvc"),
    not(target_env = "musl"),
    not(target_vendor = "apple")
))]
#[expect(
    unsafe_code,
    reason = "the allocator reads its settings from this exported symbol; a NUL-terminated byte string nothing in Rust dereferences"
)]
#[export_name = "_rjem_malloc_conf"]
pub static MALLOC_CONF: &[u8; 100] =
    b"background_thread:true,dirty_decay_ms:10000,muzzy_decay_ms:0,narenas:16,oversize_threshold:33554432\0";

/// Fills a worker before it opens the next: the worker with the most
/// requests running, by its engine's own load report, scores highest, so a
/// fleet under light traffic keeps whole workers idle. A worker with no
/// report yet counts as idle. The profile's filters say where a fill ends:
/// they drop a worker the prober takes out of rotation, and one whose
/// engine reports at or over an overload threshold.
#[derive(Debug)]
struct Pack;

/// A scorer that keeps no state takes every lifecycle hook's default.
impl Lifecycle for Pack {}

impl Scorer for Pack {
    fn name(&self) -> &'static str {
        "pack"
    }

    /// What the host must feed for this scorer to mean anything: the
    /// engines' load reports. A host that feeds a view by its policy's needs
    /// hands none to a scorer that does not declare them, and every worker
    /// then reads as idle.
    fn needs(&self) -> Needs {
        Needs {
            loads: true,
            ..Needs::default()
        }
    }

    fn score(
        &self,
        _: &RequestFacts<'_>,
        candidates: &[Candidate],
        kept: &[u32],
        out: &mut [f32],
    ) -> TieBand {
        // The load row is the engine's own report, as the store holds it.
        let running = |i: u32| {
            candidates[i as usize]
                .load
                .as_ref()
                .map_or(0, |row| row.running)
        };
        let busiest = kept.iter().map(|&i| running(i)).max().unwrap_or(0);
        for (score, &i) in out.iter_mut().zip(kept) {
            // An idle fleet ties: the picker draws among equals.
            *score = if busiest == 0 {
                1.0
            } else {
                running(i) as f32 / busiest as f32
            };
        }
        TieBand::NORMALIZED
    }
}

/// The stock build's stages and this file's scorer, under the name a policy
/// document writes.
fn stages() -> Result<Catalog, CatalogError> {
    let mut stages = Catalog::builtin();
    stages.scorer("pack", || Box::new(Pack))?;
    Ok(stages)
}

/// The crate's command line and run loop over this build's stages.
fn main() -> Result<ExitCode, CatalogError> {
    Ok(controller::run(stages()?))
}

#[cfg(test)]
mod tests {
    use std::{
        net::{Ipv4Addr, SocketAddr},
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_policy_controller::{
        controller::{Config, Controller, Election, Error, Listener, Scope, SpecRejected},
        spec::OVERLOAD_WAITING_REQUESTS,
        store::{
            Ask, GatewayId, HealthSink, LoadSink, LoadSource, MemoryStore, Overload, ViewBuffers,
            WorkerFacts, WorkerStatus, WorkersStore, PUBLISH_WINDOW,
        },
        Constraints, Engine, LoadRow, PolicySpec, RequestFacts, Role, Scorer, Scratch, SpecError,
        WorkerId,
    };

    use super::{stages, Pack};

    /// The document the example ships.
    const POLICY: &str = include_str!("../policy.yaml");

    /// The same document on disk, as `--spec-file` names it.
    const POLICY_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/policy.yaml");

    const MODEL: &str = "a-model";

    /// The queue at which a worker of a fleet of its own names itself
    /// overloaded, at its registration.
    const QUEUE: u32 = 4;

    fn read(document: &str) -> PolicySpec {
        serde_yaml::from_str(document).expect("a well-formed document")
    }

    /// The engine `document` validates into over this build's stages.
    fn engine(document: &str) -> Engine {
        let stages = stages().expect("the scorer's name is free");
        read(document)
            .validate_in(&stages, &[])
            .expect("the document names stages this build holds")
    }

    /// A store as the controller keeps one, fed through the doors its
    /// ingestion uses: two workers of one model, registered, proven ready
    /// and each with a load report on file. One runs three requests, one is
    /// idle. No row a decision reads is written by hand.
    struct Fleet {
        store: Arc<MemoryStore>,
        busy: WorkerId,
        idle: WorkerId,
        /// The overload thresholds each worker names at its registration.
        own: Overload,
        /// The instant the store's last publication window opened.
        window: Instant,
    }

    impl Fleet {
        /// A store of its own, whose workers each name a queue of [`QUEUE`]
        /// requests as their threshold.
        fn new() -> Self {
            let own = Overload {
                waiting_requests: Some(QUEUE),
                ..Overload::default()
            };
            Self::over(Arc::new(MemoryStore::new()), own)
        }

        /// The two workers on `store`, each naming `own` at its
        /// registration.
        fn over(store: Arc<MemoryStore>, own: Overload) -> Self {
            let window = Instant::now();
            let busy = Self::join(&store, window, "busy", Some(3), own);
            let idle = Self::join(&store, window, "idle", Some(0), own);
            let mut fleet = Self {
                store,
                busy,
                idle,
                own,
                window,
            };
            fleet.publish();
            fleet
        }

        /// Registers a worker and proves it ready. `running` is the load
        /// report its engine sent, and `None` an engine that has sent none.
        fn join(
            store: &MemoryStore,
            window: Instant,
            name: &str,
            running: Option<u32>,
            own: Overload,
        ) -> WorkerId {
            let key = format!("http://{name}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    models: vec![MODEL.to_owned()],
                    dp_size: 1,
                    overload: own,
                    ..WorkerFacts::default()
                })
                .id();
            let revision = WorkersStore::get(store, id).expect("the row").revision;
            store
                .set_status(id, WorkerStatus::Ready, revision)
                .expect("the prober's first pass");
            if let Some(running) = running {
                let report = LoadRow {
                    running,
                    ..LoadRow::default()
                };
                store.record(id, 0, report, window, LoadSource::Stream);
            }
            id
        }

        /// Adds a ready worker whose engine has sent no load report yet.
        fn join_unreported(&mut self, name: &str) -> WorkerId {
            let id = Self::join(&self.store, self.window, name, None, self.own);
            self.publish();
            id
        }

        /// Closes the store's publication window: a decision reads what was
        /// written before it.
        fn publish(&mut self) {
            self.window += PUBLISH_WINDOW;
            self.store.publish(self.window);
        }

        /// Moves a worker's status as the prober does.
        fn set(&mut self, id: WorkerId, status: WorkerStatus) {
            let revision = WorkersStore::get(&*self.store, id)
                .expect("the row")
                .revision;
            self.store
                .set_status(id, status, revision)
                .expect("the status moves");
            self.publish();
        }

        /// Records a load report as an engine's stream delivers one: its
        /// requests running and queued.
        fn report(&mut self, id: WorkerId, running: u32, queued: u32) {
            let report = LoadRow {
                running,
                queued,
                ..LoadRow::default()
            };
            self.store
                .record(id, 0, report, self.window, LoadSource::Stream);
            self.publish();
        }

        /// Sets or clears a gateway's standing veto on a worker, as the
        /// gateway's breaker does.
        fn veto(&mut self, gateway: GatewayId, id: WorkerId, standing: bool) {
            let until = standing.then(|| self.window + Duration::from_secs(3600));
            self.store.gateway_veto(gateway, id, until);
            self.publish();
        }

        /// What one decision asks of the store, a second into its window.
        fn ask(&self) -> Ask {
            Ask {
                gateway: GatewayId::LOCAL,
                now: self.window + Duration::from_secs(1),
                pinned: None,
                affinity: false,
            }
        }

        /// The scores `pack` writes over the pool's rows, each of them kept.
        fn scores(&self) -> Vec<f32> {
            let ask = self.ask();
            let mut view = ViewBuffers::default();
            let built = self
                .store
                .build_view(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool has members");
            let facts = RequestFacts {
                model: MODEL,
                now_micros: built.now_micros,
                ..RequestFacts::default()
            };
            let kept: Vec<u32> = (0..view.candidates.len() as u32).collect();
            let mut scores = vec![0.0; kept.len()];
            let _ = Pack.score(&facts, &view.candidates, &kept, &mut scores);
            scores
        }

        /// The first target of one decision over the store's rows.
        fn pick(&self, engine: &Engine) -> Option<WorkerId> {
            self.pick_at(engine, 0)
        }

        /// The first target of one decision by the engine the store stands
        /// under: the generation a controller applied to it.
        fn pick_standing(&self) -> Option<WorkerId> {
            let world = self.store.world();
            self.pick(world.spec().engine())
        }

        /// The same under a seed of the caller's: the picker draws among
        /// equal scores from it.
        fn pick_at(&self, engine: &Engine, seed: u64) -> Option<WorkerId> {
            let ask = self.ask();
            let mut view = ViewBuffers::default();
            let built = self
                .store
                .build_view(MODEL, Role::Single, &ask, &mut view)
                .expect("the pool has members");
            let facts = RequestFacts {
                model: MODEL,
                seed,
                now_micros: built.now_micros,
                ..RequestFacts::default()
            };
            let decision = engine.decide(
                &facts,
                &built.pool_facts(),
                &view.candidates,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            decision
                .pick()
                .and_then(|pick| pick.targets.first())
                .map(|target| target.id)
        }
    }

    #[test]
    fn the_stores_load_reports_and_health_shape_the_decision() {
        let mut fleet = Fleet::new();
        let engine = engine(POLICY);
        // The engines' own reports say which worker is busy.
        assert_eq!(fleet.pick(&engine), Some(fleet.busy));
        // The profile's filters read the store's health: a worker the
        // prober takes out of rotation is not picked, however busy.
        fleet.set(fleet.busy, WorkerStatus::NotReady);
        assert_eq!(fleet.pick(&engine), Some(fleet.idle));
    }

    #[test]
    fn a_worker_with_no_report_counts_as_idle_and_an_idle_fleet_ties() {
        let mut fleet = Fleet::new();
        let engine = engine(POLICY);
        // A third worker is ready and its engine has sent no report yet. It
        // counts as idle: beside the busy worker it is never the pick.
        let unreported = fleet.join_unreported("unreported");
        for seed in 0..32 {
            assert_eq!(fleet.pick_at(&engine, seed), Some(fleet.busy));
        }
        // The busy worker drains and nothing runs. An idle fleet ties: every
        // worker takes the same whole score, and the picker draws among
        // equals, so each of them takes picks.
        fleet.report(fleet.busy, 0, 0);
        assert_eq!(fleet.scores(), [1.0; 3]);
        let picks: Vec<_> = (0..64).map(|seed| fleet.pick_at(&engine, seed)).collect();
        for worker in [fleet.busy, fleet.idle, unreported] {
            assert!(picks.contains(&Some(worker)), "{worker:?} took no pick");
        }
    }

    #[test]
    fn a_gateways_veto_takes_its_worker_out_for_that_gateway_alone() {
        let mut fleet = Fleet::new();
        let engine = engine(POLICY);
        // Another gateway's veto is its own: this gateway still picks the
        // busy worker.
        fleet.veto(GatewayId(7), fleet.busy, true);
        assert_eq!(fleet.pick(&engine), Some(fleet.busy));
        // This gateway's veto stalls the worker for it, and the profile's
        // health filter drops a stalled worker.
        fleet.veto(GatewayId::LOCAL, fleet.busy, true);
        assert_eq!(fleet.pick(&engine), Some(fleet.idle));
        // The veto cleared, the busy worker fills again.
        fleet.veto(GatewayId::LOCAL, fleet.busy, false);
        assert_eq!(fleet.pick(&engine), Some(fleet.busy));
    }

    #[test]
    fn a_worker_at_its_overload_threshold_is_passed_over_and_the_next_opens() {
        let mut fleet = Fleet::new();
        let engine = engine(POLICY);
        // A queue under the worker's threshold: `pack` still fills it.
        fleet.report(fleet.busy, 3, QUEUE - 1);
        assert_eq!(fleet.pick(&engine), Some(fleet.busy));
        // At the threshold the store judges the worker's engine overloaded
        // and the document's `not-overloaded` drops it: the next opens.
        fleet.report(fleet.busy, 3, QUEUE);
        assert_eq!(fleet.pick(&engine), Some(fleet.idle));
        // The queue drains and the worker fills again.
        fleet.report(fleet.busy, 3, 0);
        assert_eq!(fleet.pick(&engine), Some(fleet.busy));
    }

    #[test]
    fn the_document_declares_the_load_reports_its_scorer_reads() {
        // `pack` is the document's only reader of the load reports.
        assert!(engine(POLICY).needs().loads);
    }

    #[test]
    fn the_stock_build_refuses_the_document_by_the_scorers_name() {
        match read(POLICY).validate() {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "pack"),
            other => panic!("the stock build read the document as {other:?}"),
        }
    }

    /// The crate's configuration by its fields, on loopback at ports the
    /// host picks. The command line builds the same value from its flags.
    fn config() -> Config {
        let loopback = SocketAddr::from((Ipv4Addr::LOCALHOST, 0));
        let listener = |name: &str, scopes: &[Scope]| Listener {
            name: name.to_owned(),
            addr: loopback,
            scopes: scopes.to_vec(),
            tls: None,
        };
        Config {
            controller_id: "example-0".to_owned(),
            listeners: vec![
                listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
                listener("control", &[Scope::Control]),
                listener("replication", &[Scope::Replication]),
            ],
            admin: loopback,
            decide_shards: Config::default_decide_shards(),
            shutdown_settle: Config::DEFAULT_SHUTDOWN_SETTLE,
            shutdown_grace: Config::DEFAULT_SHUTDOWN_GRACE,
            load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
            election: Election::default(),
        }
    }

    #[tokio::test]
    async fn a_controller_handed_the_document_decides_by_it_over_its_store() {
        let stages = stages().expect("the scorer's name is free");
        let mut own = Controller::bind_with(config(), stages)
            .await
            .expect("binds");
        // The start's own door: the document is judged over the stages the
        // bind was handed and applied to the controller's store.
        own.spec_file(POLICY_FILE).await.expect("the document runs");
        assert_eq!(own.store().world().spec().generation(), 1);
        // The controller's store, fed as its ingestion feeds it. Its workers
        // name no threshold of their own: the document's defaults stand.
        let mut fleet = Fleet::over(own.store(), Overload::default());
        assert_eq!(fleet.pick_standing(), Some(fleet.busy));
        // At the document's default queue the store judges the busy
        // worker's engine overloaded, and `not-overloaded` drops it.
        fleet.report(fleet.busy, 3, OVERLOAD_WAITING_REQUESTS);
        assert_eq!(fleet.pick_standing(), Some(fleet.idle));
        fleet.report(fleet.busy, 3, OVERLOAD_WAITING_REQUESTS - 1);
        assert_eq!(fleet.pick_standing(), Some(fleet.busy));
        // The stock controller refuses the same file by the scorer's name.
        let mut stock = Controller::bind(config()).await.expect("binds");
        match stock.spec_file(POLICY_FILE).await {
            Err(Error::SpecFile {
                source: SpecRejected::Invalid(SpecError::UnknownScorer { name, .. }),
                ..
            }) => assert_eq!(name, "pack"),
            other => panic!("the stock controller read the file as {other:?}"),
        }
    }

    #[tokio::test]
    async fn a_controller_bound_with_these_stages_judges_the_document_by_them() {
        let stages = stages().expect("the scorer's name is free");
        let own = Controller::bind_with(config(), stages)
            .await
            .expect("binds");
        assert!(read(POLICY).validate_in(own.catalog(), &[]).is_ok());
        // A controller bound with the stock stages refuses the same document
        // by the scorer's name.
        let stock = Controller::bind(config()).await.expect("binds");
        match read(POLICY).validate_in(stock.catalog(), &[]) {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "pack"),
            other => panic!("the stock controller read the document as {other:?}"),
        }
    }
}
