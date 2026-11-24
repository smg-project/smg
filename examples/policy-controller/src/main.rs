//! A policy controller of your own: the stock controller with one scorer
//! added in code. The scorer is a type of this file; the built-in stages,
//! the command line, the listeners and the election are the crate's.
//!
//! A change to the view a stage reads ([`Candidate`], [`RequestFacts`]) is a
//! change to what this file compiles against, as it is for a plugin's world.

use std::process::ExitCode;

use smg_policy_controller::{
    document::Needs,
    engine::{Candidate, RequestFacts, Scorer, TieBand},
    service::controller,
    stages::{Lifecycle, StageRegistry, StageRegistryError},
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
        Needs::NONE.with_loads()
    }

    /// This scorer only places, and by the engines' own reports: it turns
    /// no request away. Saying so leaves a price: a stage that says
    /// nothing is taken as one that may refuse a request by what the
    /// gateways report of leases, and each decision under it applies those
    /// reports first.
    fn refuses_by_reports(&self) -> bool {
        false
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
fn stages() -> Result<StageRegistry, StageRegistryError> {
    let mut stages = StageRegistry::builtin();
    stages.scorer("pack", || Box::new(Pack))?;
    Ok(stages)
}

/// The crate's command line and run loop over this build's stages.
fn main() -> Result<ExitCode, StageRegistryError> {
    Ok(controller::run(stages()?))
}

#[cfg(test)]
mod tests {
    use std::{
        collections::HashMap,
        future::Future,
        net::{Ipv4Addr, SocketAddr},
        sync::Arc,
        time::Duration,
    };

    use smg_policy_controller::{
        document::{operator::SpecRejected, AppliedSpec, PolicySpec, SpecError},
        engine::{
            Candidate, LentList, ListCell, LoadRow, PlaceRequest, PolicyEngine, RequestFacts,
            Scorer, TieBand, Unplaced, WorkerLists, WorkerRule,
        },
        service::controller::{Config, Controller, ElectionConfig, Error, Listener, Scope},
        stages::StageRegistry,
    };
    use smg_worker::{
        identity::basic::Worker,
        load::state::{one_rank_load, TestFeed},
        test_support::ready_worker,
        worker::{WorkerLoadResponse, WorkerStatus},
    };

    use super::{stages, Pack};

    /// The document the example ships.
    const POLICY: &str = include_str!("../policy.yaml");

    /// The same document on disk, as `--spec-file` names it.
    const POLICY_FILE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/policy.yaml");

    const MODEL: &str = "a-model";

    fn read(document: &str) -> PolicySpec {
        serde_yaml::from_str(document).expect("a well-formed document")
    }

    /// The generation `document` validates into over `stages`.
    fn applied_in(document: &str, stages: &StageRegistry) -> AppliedSpec {
        AppliedSpec::validate_in(&read(document), stages, &[])
            .expect("the document names stages this build holds")
    }

    /// The generation `document` validates into over this build's stages.
    fn applied(document: &str) -> AppliedSpec {
        applied_in(document, &stages().expect("the scorer's name is free"))
    }

    /// A worker list as a controller holds one: two workers of one model,
    /// the worker crate's own worker objects, each ready, in the list a
    /// registry publishes with the cell its holder keeps beside it. A load
    /// feed stands where the load monitor does, and the engine is handed
    /// its channel once, at the start. One worker runs three requests, one
    /// is idle. No row a decision reads is written by hand: the engine
    /// writes each from a worker's object and its engine's report.
    struct Fleet {
        /// The engine, built for the life of the process: a placement holds
        /// it for as long as its request runs. The fixture makes each
        /// hand-over of a program itself: the load channel once at the
        /// start, and a worker's address as it joins.
        engine: &'static PolicyEngine,
        workers: Arc<[Arc<dyn Worker>]>,
        cell: ListCell,
        loads: TestFeed,
        /// Each worker's newest load report, by its address.
        reports: HashMap<String, WorkerLoadResponse>,
        busy: Arc<dyn Worker>,
        idle: Arc<dyn Worker>,
    }

    impl Fleet {
        /// The two workers under the document the example ships.
        fn new() -> Self {
            Self::under(applied(POLICY))
        }

        /// The two workers under `spec`.
        fn under(spec: AppliedSpec) -> Self {
            let engine: &'static PolicyEngine = Box::leak(Box::new(PolicyEngine::new(spec)));
            let loads = TestFeed::attached_to(|channel| {
                if let Some(channel) = channel {
                    engine.set_loads(channel);
                }
            });
            let busy = ready_worker("http://busy:8000", MODEL);
            let idle = ready_worker("http://idle:8000", MODEL);
            let mut fleet = Self {
                engine,
                workers: Vec::new().into(),
                cell: ListCell::default(),
                loads,
                reports: HashMap::new(),
                busy: Arc::clone(&busy),
                idle: Arc::clone(&idle),
            };
            fleet.join(busy, Some(3));
            fleet.join(idle, Some(0));
            fleet
        }

        /// A worker joins: the list is published again with an empty cell
        /// beside it, as a registry publishes a membership generation, and
        /// the engine is told. `running` is the load report the worker's
        /// engine sent, and `None` an engine that has sent none.
        fn join(&mut self, worker: Arc<dyn Worker>, running: Option<i32>) {
            let listed = self.workers.iter().cloned();
            self.workers = listed.chain([Arc::clone(&worker)]).collect();
            self.cell = ListCell::default();
            self.engine.joined(worker.url());
            if let Some(running) = running {
                self.report(&worker, running);
            }
        }

        /// Publishes a load report as a worker's engine sends one: its
        /// requests running.
        fn report(&mut self, worker: &Arc<dyn Worker>, running: i32) {
            let mut report = one_rank_load(0, 0.0, 0.0);
            for rank in &mut report.loads {
                rank.num_running_reqs = running;
            }
            self.reports.insert(worker.url().to_owned(), report);
            self.loads.publish(&self.reports);
        }

        /// The address of the worker one request is placed on: the engine's
        /// one call for a request, over the list as it stands.
        fn pick(&self) -> Option<String> {
            self.pick_where(None)
        }

        /// Whether eight requests in a row are placed on `worker`. Among
        /// equal scores the picker's draw moves from request to request,
        /// so a fleet that ties does not answer one worker eight times.
        fn fills(&self, worker: &Arc<dyn Worker>) -> bool {
            (0..8).all(|_| self.pick().as_deref() == Some(worker.url()))
        }

        /// [`Self::pick`] for a request whose route states a rule of its
        /// own for a worker it may be sent to.
        fn pick_where(&self, admits: Option<WorkerRule<'_>>) -> Option<String> {
            self.place(admits).ok()
        }

        /// The engine's one call for a request over the list as it stands:
        /// the address of the worker it is placed on, or why it has none.
        fn place(&self, admits: Option<WorkerRule<'_>>) -> Result<String, Unplaced> {
            let lent = LentList {
                workers: &self.workers,
                cell: &self.cell,
                facts: None,
            };
            let request = PlaceRequest {
                admits,
                ..PlaceRequest::of(MODEL, WorkerLists::Single(lent))
            };
            let placement = self.engine.place(&request)?;
            Ok(placement.worker.url().to_owned())
        }
    }

    /// `case` beside the engine's own task, as a program runs it: the task
    /// keeps the newest load snapshot at hand. It runs for as long as its
    /// engine stands, so the case's end ends both.
    ///
    /// The task is polled ahead of the case at every turn. Its first tick
    /// is due before the case's first wait ends, and a report wakes it at
    /// once, so a report the feed published is at hand before the case
    /// reads on, on a paused clock and on a real one.
    async fn beside_its_task(engine: &'static PolicyEngine, case: impl Future<Output = ()>) {
        tokio::select! {
            biased;
            () = engine.run() => {}
            () = case => {}
        }
    }

    /// Long enough for the engine's task to take a report the feed just
    /// published.
    async fn seen() {
        tokio::time::sleep(Duration::from_millis(1)).await;
    }

    #[tokio::test(start_paused = true)]
    async fn the_workers_load_reports_and_health_shape_the_decision() {
        let fleet = Fleet::new();
        let case = async {
            seen().await;
            // The engines' own reports say which worker is busy.
            assert!(fleet.fills(&fleet.busy));
            // A worker taken out of rotation is not picked, however busy.
            fleet.busy.set_status(WorkerStatus::NotReady);
            assert_eq!(fleet.pick().as_deref(), Some(fleet.idle.url()));
            // With no worker in rotation the request is shed, and by the
            // document's own filter: the profile lists `healthy`, so its
            // decision holds no worker that is out of rotation. A profile
            // that listed no such filter would name one, and the call would
            // answer that it has no worker to send to.
            fleet.idle.set_status(WorkerStatus::NotReady);
            match fleet.place(None) {
                Err(Unplaced::Shed { emptied_by, .. }) => assert_eq!(emptied_by, "healthy"),
                other => panic!("the request was answered {other:?}"),
            }
        };
        beside_its_task(fleet.engine, case).await;
    }

    /// The scorer alone, over rows made by hand. A worker scores its share
    /// of the busiest worker's running requests. Where nothing runs no
    /// worker is the busiest, and each takes the whole score: an idle fleet
    /// ties.
    #[test]
    fn the_scorer_scores_a_workers_share_of_the_busiest_and_ties_an_idle_fleet() {
        let row = |running: Option<u32>| Candidate {
            load: running.map(|running| LoadRow {
                running,
                ..LoadRow::default()
            }),
            ..Candidate::default()
        };
        let scores = |rows: &[Candidate]| {
            let kept: Vec<u32> = (0..).take(rows.len()).collect();
            let mut out = vec![0.0_f32; rows.len()];
            let band = Pack.score(&RequestFacts::default(), rows, &kept, &mut out);
            assert_eq!(band, TieBand::NORMALIZED);
            out
        };
        // A worker with no report counts as idle beside the busy ones.
        let busy = [row(Some(4)), row(Some(1)), row(None)];
        assert_eq!(scores(&busy), [1.0, 0.25, 0.0]);
        let idle = [row(Some(0)), row(None), row(Some(0))];
        assert_eq!(scores(&idle), [1.0, 1.0, 1.0]);
    }

    #[tokio::test(start_paused = true)]
    async fn a_worker_that_joins_late_is_in_force_at_once() {
        let mut fleet = Fleet::new();
        let engine = fleet.engine;
        let case = async {
            seen().await;
            assert!(fleet.fills(&fleet.busy));
            // A worker joins behind the first decision, running more
            // requests than any other: the next request fills it.
            let late = ready_worker("http://late:8000", MODEL);
            fleet.join(Arc::clone(&late), Some(5));
            seen().await;
            assert!(fleet.fills(&late));
        };
        beside_its_task(engine, case).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_worker_with_no_report_counts_as_idle_and_an_idle_fleet_ties() {
        let mut fleet = Fleet::new();
        let engine = fleet.engine;
        let case = async {
            // A third worker is ready and its engine has sent no report
            // yet. It counts as idle: beside the busy worker it is never
            // the pick.
            let unreported = ready_worker("http://unreported:8000", MODEL);
            fleet.join(Arc::clone(&unreported), None);
            seen().await;
            assert!(fleet.fills(&fleet.busy));
            // The busy worker drains and nothing runs. An idle fleet ties:
            // every worker takes the same whole score, and the picker
            // draws among equals, so each of them takes picks.
            let busy = Arc::clone(&fleet.busy);
            fleet.report(&busy, 0);
            seen().await;
            let picks: Vec<_> = (0..64).map(|_| fleet.pick()).collect();
            for worker in [&fleet.busy, &fleet.idle, &unreported] {
                let address = Some(worker.url().to_owned());
                assert!(picks.contains(&address), "{} took no pick", worker.url());
            }
        };
        beside_its_task(engine, case).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_routes_own_rule_takes_a_worker_out_for_its_requests_alone() {
        let fleet = Fleet::new();
        let case = async {
            seen().await;
            // A request whose route states no rule fills the busy worker.
            assert!(fleet.fills(&fleet.busy));
            // A route's own rule refuses the busy worker: its request
            // opens the next worker.
            let refused = fleet.busy.url().to_owned();
            let rule = |worker: &dyn Worker| worker.url() != refused;
            let ruled = fleet.pick_where(Some(&rule));
            assert_eq!(ruled.as_deref(), Some(fleet.idle.url()));
            // The rule is that request's own: the next request, which
            // states none, fills the busy worker again.
            assert!(fleet.fills(&fleet.busy));
        };
        beside_its_task(fleet.engine, case).await;
    }

    #[tokio::test(start_paused = true)]
    async fn a_worker_judged_overloaded_is_passed_over_and_the_next_opens() {
        let fleet = Fleet::new();
        let case = async {
            seen().await;
            assert!(fleet.fills(&fleet.busy));
            // The worker's host judges its engine at an overload threshold,
            // and the document's `not-overloaded` drops it: the next opens.
            fleet.busy.set_overloaded(true);
            assert_eq!(fleet.pick().as_deref(), Some(fleet.idle.url()));
            // The verdict is lifted and the worker fills again.
            fleet.busy.set_overloaded(false);
            assert!(fleet.fills(&fleet.busy));
        };
        beside_its_task(fleet.engine, case).await;
    }

    #[test]
    fn the_document_declares_the_load_reports_its_scorer_reads() {
        // `pack` is the document's only reader of the load reports.
        assert!(applied(POLICY).needs().loads);
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
            election: ElectionConfig::default(),
            discovery: None,
        }
    }

    #[tokio::test]
    async fn a_controller_handed_the_document_runs_it_and_it_decides_over_a_worker_list() {
        let stages = stages().expect("the scorer's name is free");
        let mut own = Controller::bind_with(config(), stages)
            .await
            .expect("binds");
        // The start's own door: the document is judged over the stages the
        // bind was handed, and the controller runs it as its generation 1.
        own.spec_file(POLICY_FILE).await.expect("the document runs");
        assert_eq!(own.store().snapshot().spec().generation(), 1);
        // The document as the controller's own stages judge it, deciding
        // over a worker list.
        let fleet = Fleet::under(applied_in(POLICY, own.stage_registry()));
        let case = async {
            seen().await;
            assert!(fleet.fills(&fleet.busy));
            fleet.busy.set_overloaded(true);
            assert_eq!(fleet.pick().as_deref(), Some(fleet.idle.url()));
        };
        beside_its_task(fleet.engine, case).await;
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
        assert!(read(POLICY).validate_in(own.stage_registry(), &[]).is_ok());
        // A controller bound with the stock stages refuses the same document
        // by the scorer's name.
        let stock = Controller::bind(config()).await.expect("binds");
        match read(POLICY).validate_in(stock.stage_registry(), &[]) {
            Err(SpecError::UnknownScorer { name, .. }) => assert_eq!(name, "pack"),
            other => panic!("the stock controller read the document as {other:?}"),
        }
    }
}
