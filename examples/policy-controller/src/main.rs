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

/// Fills a worker before it opens the next: the worker with the most
/// requests running, by its engine's own load report, scores highest, so a
/// fleet under light traffic keeps whole workers idle. A worker with no
/// report yet counts as idle. The profile's filters still drop a worker
/// that is unhealthy or overloaded.
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
        time::{Duration, Instant},
    };

    use smg_policy_controller::{
        controller::{Config, Controller, Election, Listener, Scope},
        store::{
            Ask, GatewayId, LoadSink, LoadSource, MemoryStore, ViewBuffers, WorkerFacts,
            WorkerStatus, WorkersStore, PUBLISH_WINDOW,
        },
        Constraints, Engine, LoadRow, PolicySpec, RequestFacts, Role, Scorer, Scratch, SpecError,
        WorkerId,
    };

    use super::{stages, Pack};

    /// The document the example ships.
    const POLICY: &str = include_str!("../policy.yaml");

    const MODEL: &str = "a-model";

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
        store: MemoryStore,
        busy: WorkerId,
        idle: WorkerId,
        /// The instant the store's last publication window opened.
        window: Instant,
    }

    impl Fleet {
        fn new() -> Self {
            let store = MemoryStore::new();
            let window = Instant::now();
            let busy = Self::join(&store, window, "busy", Some(3));
            let idle = Self::join(&store, window, "idle", Some(0));
            let mut fleet = Self {
                store,
                busy,
                idle,
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
        ) -> WorkerId {
            let key = format!("http://{name}:8000");
            let id = store
                .upsert(WorkerFacts {
                    key: key.clone(),
                    engine: key,
                    incarnation: 1,
                    models: vec![MODEL.to_owned()],
                    dp_size: 1,
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
            let id = Self::join(&self.store, self.window, name, None);
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
            let revision = WorkersStore::get(&self.store, id)
                .expect("the row")
                .revision;
            self.store
                .set_status(id, status, revision)
                .expect("the status moves");
            self.publish();
        }

        /// Records a load report as an engine's stream delivers one.
        fn report(&mut self, id: WorkerId, running: u32) {
            let report = LoadRow {
                running,
                ..LoadRow::default()
            };
            self.store
                .record(id, 0, report, self.window, LoadSource::Stream);
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
        fleet.report(fleet.busy, 0);
        assert_eq!(fleet.scores(), [1.0; 3]);
        let picks: Vec<_> = (0..64).map(|seed| fleet.pick_at(&engine, seed)).collect();
        for worker in [fleet.busy, fleet.idle, unreported] {
            assert!(picks.contains(&Some(worker)), "{worker:?} took no pick");
        }
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
            shutdown_settle: Config::DEFAULT_SHUTDOWN_SETTLE,
            shutdown_grace: Config::DEFAULT_SHUTDOWN_GRACE,
            load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
            election: Election::default(),
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
