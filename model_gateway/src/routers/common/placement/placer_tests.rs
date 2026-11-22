//! The one placing call, asked of both sides over the same pool.
//!
//! [`Placer::place`] is one pass for the classic policies and for the policy
//! engine: the pool on its wire, the request's own rule, the engines a retry
//! tried, then whoever decides. Each line of the table below states one
//! attempt, runs it through both sides, and reads the same workers taken and
//! the same miss back. A step that moves a route onto the call adds its
//! lines here. [`Placer::place_from`] is the same pass over a list its route
//! made, and its lines read the same way.

use std::collections::BTreeSet;

use metrics_exporter_prometheus::PrometheusBuilder;
use openai_protocol::{model_card::ModelCard, worker::HealthCheckConfig};
use smg_worker::{
    identity::basic::{BasicWorker, WorkerLoadGuard},
    UNKNOWN_MODEL_ID,
};

use super::*;
use crate::{
    config::{types::PolicyConfig, RouterConfig},
    policy_controller::start_in_process,
    routers::grpc::backend_client::BasicWorkerBuilder,
};

const MODEL: &str = "m";

/// The times one attempt is put to a side: round robin takes each candidate
/// in turn, so the workers a side placed on are the workers it was handed.
const TURNS: usize = 12;

/// Both sides over one worker registry, each built from one configuration
/// as a gateway's start builds it: the classic policies of the routing
/// settings, and the engine of this process over the document the same
/// settings write, with the registry its routers hold beside it.
struct Sides {
    registry: Arc<WorkerRegistry>,
    classic: Placer,
    engine: Placer,
}

impl Sides {
    /// The sides of a round robin gateway over `workers`, which sheds a
    /// pool over its thresholds where `shed`.
    fn over(workers: Vec<BasicWorker>, shed: bool) -> Self {
        Self::labelled(workers, shed, None)
    }

    /// [`Self::over`], with the model's own policy as a worker's label
    /// names it.
    fn labelled(workers: Vec<BasicWorker>, shed: bool, label: Option<&str>) -> Self {
        let registry = Arc::new(WorkerRegistry::new());
        registry.set_overload_shed(shed);
        let config = RouterConfig {
            policy: PolicyConfig::RoundRobin,
            policy_controller: true,
            worker_overload_shed: shed,
            ..RouterConfig::default()
        };
        let classic = Arc::new(PolicyRegistry::with_override(
            config.policy.clone(),
            config.routing_key_override.clone(),
        ));
        let started = start_in_process(&config, None)
            .expect("the settings' document starts")
            .expect("round robin is placed by a policy");
        assert!(started.decides_whole(), "the engine decides round robin");
        let object = Arc::new(PolicyEngine::own(
            Arc::clone(&started.engine),
            Arc::clone(&registry),
        ));
        let beside = Arc::new(PolicyRegistry::with_policy_engine(
            config.policy.clone(),
            config.routing_key_override.clone(),
            Arc::clone(&started.engine),
            started.origin,
            true,
        ));
        for worker in workers {
            let model = worker.model_id().to_owned();
            registry
                .register(Arc::new(worker))
                .expect("the worker registers");
            classic.on_worker_added(&model, label);
            beside.on_worker_added(&model, label);
        }
        Self {
            registry,
            classic: Placer::of(&classic, None),
            engine: Placer::of(&beside, Some(&object)),
        }
    }

    /// One attempt put to both sides, [`TURNS`] times each: what each did
    /// with it, which the line then reads. The two must agree.
    #[track_caller]
    fn agree(
        &self,
        line: &str,
        model: &str,
        pool: RoutingPool,
        wire: Option<WireConstraint>,
        inputs: &PlacementInputs<'_>,
    ) -> Seen {
        let classic = Seen::of(&self.classic, &self.registry, model, pool, wire, inputs);
        let engine = Seen::of(&self.engine, &self.registry, model, pool, wire, inputs);
        assert_eq!(classic, engine, "{line}: the classic side, then the engine");
        classic
    }

    /// [`Self::agree`] for an attempt placed over `candidates`, a list its
    /// route made of `pool` ([`Placer::place_from`]).
    #[track_caller]
    fn agree_from(
        &self,
        line: &str,
        pool: RoutingPool,
        candidates: &[Arc<dyn Worker>],
        inputs: &PlacementInputs<'_>,
    ) -> Seen {
        let over = |placer: &Placer| {
            Seen::repeated(|| placer.place_from(&self.registry, MODEL, pool, candidates, inputs))
        };
        let (classic, engine) = (over(&self.classic), over(&self.engine));
        assert_eq!(classic, engine, "{line}: the classic side, then the engine");
        classic
    }
}

/// What a side did with one attempt, asked several times: the workers it
/// placed the request on, and whether it placed nobody.
#[derive(Debug, PartialEq, Eq)]
struct Seen {
    placed: BTreeSet<String>,
    missed: bool,
}

impl Seen {
    fn of(
        placer: &Placer,
        registry: &WorkerRegistry,
        model: &str,
        pool: RoutingPool,
        wire: Option<WireConstraint>,
        inputs: &PlacementInputs<'_>,
    ) -> Self {
        Self::repeated(|| placer.place(registry, model, pool, wire, inputs))
    }

    /// What `place`, one attempt put to one side, did when it was
    /// repeated [`TURNS`] times.
    fn repeated(place: impl Fn() -> Result<Placed, Missed>) -> Self {
        let mut seen = Self {
            placed: BTreeSet::new(),
            missed: false,
        };
        for _ in 0..TURNS {
            match place() {
                Ok(placed) => {
                    seen.placed.insert(placed.worker().url().to_owned());
                }
                Err(Missed::Nobody) => seen.missed = true,
                Err(Missed::Engine(verdict)) => {
                    panic!("a verdict of the engine's own: {verdict:?}")
                }
            }
        }
        seen
    }

    /// The side placed every turn, on exactly `urls`.
    #[track_caller]
    fn placed_on(&self, line: &str, urls: &[&str]) {
        let urls: BTreeSet<String> = urls.iter().map(|url| (*url).to_owned()).collect();
        assert_eq!(self.placed, urls, "{line}");
        assert!(!self.missed, "{line}: no turn placed nobody");
    }

    /// The side placed nobody, at every turn.
    #[track_caller]
    fn placed_nobody(&self, line: &str) {
        assert!(self.missed && self.placed.is_empty(), "{line}: {self:?}");
    }
}

fn worker(url: &str, model: &str, connection: ConnectionMode, runtime: RuntimeType) -> BasicWorker {
    BasicWorkerBuilder::new(url)
        .model(ModelCard::new(model))
        .worker_type(WorkerType::Regular)
        .connection_mode(connection)
        .runtime_type(runtime)
        .health_config(HealthCheckConfig {
            disable_health_check: true,
            ..Default::default()
        })
        .build()
}

fn http(url: &str) -> BasicWorker {
    worker(url, MODEL, ConnectionMode::Http, RuntimeType::Sglang)
}

fn three() -> Vec<BasicWorker> {
    vec![http("http://h:1"), http("http://h:2"), http("http://h:3")]
}

fn target(index: usize) -> HeaderMap {
    let mut headers = HeaderMap::new();
    headers.insert(
        "x-smg-target-worker",
        HeaderValue::from_str(&index.to_string()).expect("a number"),
    );
    headers
}

/// Both sides are handed the same workers and hand back the same miss, for
/// each rule of the pass.
#[test]
fn both_sides_place_over_the_same_candidates() {
    const HTTP: RoutingPool = RoutingPool::HttpRegular;
    let plain = PlacementInputs::default();

    let line = "the pool whole";
    let sides = Sides::over(three(), false);
    sides
        .agree(line, MODEL, HTTP, None, &plain)
        .placed_on(line, &["http://h:1", "http://h:2", "http://h:3"]);

    let line = "a wire pin keeps the workers on the pinned runtime and transport";
    let mixed = Sides::over(
        vec![
            worker("grpc://v:1", MODEL, ConnectionMode::Grpc, RuntimeType::Vllm),
            worker(
                "grpc://s:1",
                MODEL,
                ConnectionMode::Grpc,
                RuntimeType::Sglang,
            ),
            worker("grpc://v:2", MODEL, ConnectionMode::Grpc, RuntimeType::Vllm),
        ],
        false,
    );
    let pinned = WireConstraint {
        runtime: RuntimeType::Sglang,
        connection: ConnectionMode::Grpc,
        requires_media_refs: false,
        pipeline: PipelinePin::Any,
    };
    mixed
        .agree(
            line,
            MODEL,
            RoutingPool::GrpcPipelineRegular,
            Some(pinned),
            &plain,
        )
        .placed_on(line, &["grpc://s:1"]);

    let line = "the request's own rule keeps a worker out";
    let rule = |worker: &dyn Worker| worker.url() != "http://h:2";
    let ruled = PlacementInputs {
        candidate_filter: Some(&rule),
        ..Default::default()
    };
    sides
        .agree(line, MODEL, HTTP, None, &ruled)
        .placed_on(line, &["http://h:1", "http://h:3"]);

    let line = "a rule that refuses every worker places nobody";
    let nobody = |_: &dyn Worker| false;
    let refused = PlacementInputs {
        candidate_filter: Some(&nobody),
        ..Default::default()
    };
    sides
        .agree(line, MODEL, HTTP, None, &refused)
        .placed_nobody(line);

    let line = "a named worker holds its request, and its retry";
    let hashed = Sides::labelled(three(), false, Some("consistent_hashing"));
    let headers = target(1);
    let named = PlacementInputs {
        headers: Some(&headers),
        ..Default::default()
    };
    let held = hashed.agree(line, MODEL, HTTP, None, &named);
    assert_eq!(held.placed.len(), 1, "{line}: {held:?}");
    assert!(!held.missed, "{line}");
    let tried: Vec<String> = held.placed.iter().cloned().collect();
    let again = PlacementInputs {
        tried: &tried,
        attempt: 1,
        ..named
    };
    assert_eq!(
        hashed.agree(line, MODEL, HTTP, None, &again),
        held,
        "{line}"
    );

    let line = "a request that names no worker of the pool moves to no other";
    let headers = target(7);
    let astray = PlacementInputs {
        headers: Some(&headers),
        ..Default::default()
    };
    hashed
        .agree(line, MODEL, HTTP, None, &astray)
        .placed_nobody(line);

    let line = "every rank of a tried engine steps aside";
    let rank = |rank: usize| {
        BasicWorkerBuilder::new("http://e:8000")
            .model(ModelCard::new(MODEL))
            .dp_config(rank, 2)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build()
    };
    let ranked = Sides::over(vec![rank(0), rank(1), http("http://o:8000")], false);
    let tried = vec!["http://e:8000".to_owned()];
    let retried = PlacementInputs {
        tried: &tried,
        attempt: 1,
        ..Default::default()
    };
    ranked
        .agree(line, MODEL, HTTP, None, &retried)
        .placed_on(line, &["http://o:8000"]);

    let line = "a pool whose every engine was tried is used whole";
    let tried = vec!["http://e:8000".to_owned(), "http://o:8000".to_owned()];
    let spent = PlacementInputs {
        tried: &tried,
        attempt: 2,
        ..Default::default()
    };
    let whole = ranked.agree(line, MODEL, HTTP, None, &spent);
    assert_eq!(whole.placed.len(), 3, "{line}: {whole:?}");
    assert!(!whole.missed, "{line}");

    let line = "a model no worker serves places nobody";
    sides
        .agree(line, "absent", HTTP, None, &plain)
        .placed_nobody(line);
    assert!(
        matches!(
            single_failure(&sides.registry, "absent", HTTP, None),
            PlacementFailure::NoCandidates
        ),
        "{line}: the route's judgment is the model not found"
    );

    let line = "a request that names no model is placed over every worker";
    let models = Sides::over(
        vec![
            worker("http://m:1", "m", ConnectionMode::Http, RuntimeType::Sglang),
            worker("http://n:1", "n", ConnectionMode::Http, RuntimeType::Sglang),
        ],
        false,
    );
    models
        .agree(line, UNKNOWN_MODEL_ID, HTTP, None, &plain)
        .placed_on(line, &["http://m:1", "http://n:1"]);
}

/// A pool whose every worker is over its thresholds: each side steers by
/// its own rule where the gateway does not shed, and both place nobody
/// where it does, so the route's one judgment builds the shed.
#[test]
fn a_pool_over_its_thresholds_is_steered_by_each_sides_own_rule() {
    const HTTP: RoutingPool = RoutingPool::HttpRegular;
    let plain = PlacementInputs::default();
    let overloaded = || {
        let workers = three();
        for worker in &workers {
            worker.set_overloaded(true);
        }
        workers
    };

    let steering = Sides::over(overloaded(), false);
    // The second worker holds the fewest requests in flight.
    let pool = steering.registry.get_routing_pool(MODEL, HTTP);
    let _loads: Vec<WorkerLoadGuard> = pool
        .iter()
        .flat_map(|worker| {
            let load = match worker.url() {
                "http://h:1" => 3,
                "http://h:2" => 1,
                _ => 2,
            };
            (0..load).map(|_| WorkerLoadGuard::with_key(Arc::clone(worker), None))
        })
        .collect();
    let classic = Seen::of(
        &steering.classic,
        &steering.registry,
        MODEL,
        HTTP,
        None,
        &plain,
    );
    let engine = Seen::of(
        &steering.engine,
        &steering.registry,
        MODEL,
        HTTP,
        None,
        &plain,
    );
    classic.placed_on(
        "the classic default takes the least-loaded worker",
        &["http://h:2"],
    );
    engine.placed_on(
        "the engine's rule takes the least-vetoed worker",
        &["http://h:2"],
    );

    let shedding = Sides::over(overloaded(), true);
    let line = "under the shed setting both sides place nobody";
    shedding
        .agree(line, MODEL, HTTP, None, &plain)
        .placed_nobody(line);
    assert!(
        matches!(
            single_failure(&shedding.registry, MODEL, HTTP, None),
            PlacementFailure::AllOverloaded(_)
        ),
        "{line}: the route's judgment is the shed"
    );
}

/// Each side counts its selection under the name of the routing settings'
/// policy, and counts a pool over its thresholds that it steered.
#[test]
fn both_sides_count_a_selection_and_the_steer_of_a_vetoed_pool() {
    let plain = PlacementInputs::default();
    // One request placed by `placer`, and what the placement counted.
    let counted = |placer: &Placer, registry: &WorkerRegistry| {
        let recorder = PrometheusBuilder::new().build_recorder();
        let placed = {
            let _guard = metrics::set_default_local_recorder(&recorder);
            placer.place(registry, MODEL, RoutingPool::HttpRegular, None, &plain)
        };
        assert!(placed.is_ok(), "a worker takes the request");
        recorder.handle().render()
    };
    let once = |rendered: &str, series: &str, labels: &[&str]| {
        rendered.lines().any(|line| {
            line.starts_with(series)
                && labels.iter().all(|label| line.contains(label))
                && line.ends_with(" 1")
        })
    };
    const SELECTED: &[&str] = &[r#"policy="round_robin""#, r#"connection_mode="http""#];
    const STEERED: &[&str] = &[r#"stage="selection""#];

    let sides = Sides::over(three(), false);
    for (side, placer) in [("classic", &sides.classic), ("engine", &sides.engine)] {
        let rendered = counted(placer, &sides.registry);
        assert!(
            once(&rendered, "smg_worker_selection_total{", SELECTED),
            "{side}: {rendered}"
        );
        assert!(
            !rendered.contains("smg_worker_overload_fallback_total"),
            "{side}: no pool was steered: {rendered}"
        );
    }

    let workers = three();
    for worker in &workers {
        worker.set_overloaded(true);
    }
    let steering = Sides::over(workers, false);
    for (side, placer) in [("classic", &steering.classic), ("engine", &steering.engine)] {
        let rendered = counted(placer, &steering.registry);
        assert!(
            once(&rendered, "smg_worker_overload_fallback_total{", STEERED),
            "{side}: {rendered}"
        );
        assert!(
            once(&rendered, "smg_worker_selection_total{", SELECTED),
            "{side}: {rendered}"
        );
    }
}

/// A route that makes its own list places over it through both sides
/// ([`Placer::place_from`]): the transcription route's list is its pool less
/// the workers that registered a data-parallel rank. Both sides are handed
/// that list and no other worker of the pool, and a miss is the route's to
/// judge over the list.
#[test]
fn both_sides_place_over_a_list_its_route_made() {
    const HTTP: RoutingPool = RoutingPool::HttpRegular;
    let plain = PlacementInputs::default();
    let ranked = |rank: usize| {
        BasicWorkerBuilder::new("http://e:8000")
            .model(ModelCard::new(MODEL))
            .dp_config(rank, 2)
            .health_config(HealthCheckConfig {
                disable_health_check: true,
                ..Default::default()
            })
            .build()
    };
    let unranked = |sides: &Sides| -> Vec<Arc<dyn Worker>> {
        sides
            .registry
            .get_routing_pool(MODEL, HTTP)
            .iter()
            .filter(|worker| !worker.is_dp_aware())
            .cloned()
            .collect()
    };

    let line = "a list keeps the pool's other workers out";
    let sides = Sides::over(
        vec![ranked(0), ranked(1), http("http://h:1"), http("http://h:2")],
        false,
    );
    let list = unranked(&sides);
    sides
        .agree_from(line, HTTP, &list, &plain)
        .placed_on(line, &["http://h:1", "http://h:2"]);

    let line = "the request's own rule narrows a list too";
    let rule = |worker: &dyn Worker| worker.url() != "http://h:2";
    let ruled = PlacementInputs {
        candidate_filter: Some(&rule),
        ..Default::default()
    };
    sides
        .agree_from(line, HTTP, &list, &ruled)
        .placed_on(line, &["http://h:1"]);

    let line = "a list with no worker places nobody";
    sides
        .agree_from(line, HTTP, &[], &plain)
        .placed_nobody(line);

    // One worker with no rank, over its thresholds, beside a free worker
    // that registered a rank.
    let over_beside_ranked = || {
        let over = http("http://h:1");
        over.set_overloaded(true);
        vec![ranked(0), over]
    };

    let line = "a list over its thresholds is steered within the list";
    let steering = Sides::over(over_beside_ranked(), false);
    let list = unranked(&steering);
    steering
        .agree_from(line, HTTP, &list, &plain)
        .placed_on(line, &["http://h:1"]);

    let line = "a list over its thresholds places nobody where the gateway sheds";
    let shedding = Sides::over(over_beside_ranked(), true);
    let list = unranked(&shedding);
    shedding
        .agree_from(line, HTTP, &list, &plain)
        .placed_nobody(line);
    assert!(
        matches!(
            failure_from(&list, MODEL, true),
            PlacementFailure::AllOverloaded(_)
        ),
        "{line}: the route's judgment over its list is the shed"
    );
    assert!(
        matches!(
            single_failure(&shedding.registry, MODEL, HTTP, None),
            PlacementFailure::Unavailable
        ),
        "{line}: the pool whole holds a free worker, and is not shed"
    );
}

/// The attempt of a request whose body is sent on unread: it names no model
/// and states the tokens of its hint as its prompt. Both sides place it over
/// every worker.
#[test]
fn both_sides_place_an_unread_body_over_every_worker() {
    let sides = Sides::over(
        vec![
            worker("http://m:1", "m", ConnectionMode::Http, RuntimeType::Sglang),
            worker("http://n:1", "n", ConnectionMode::Http, RuntimeType::Sglang),
        ],
        false,
    );
    let hint: Vec<u32> = (1..=64).collect();
    let headers = HeaderMap::new();
    let hinted = PlacementInputs {
        tokens: Some(&hint),
        headers: Some(&headers),
        ..Default::default()
    };
    let line = "no model named, the hint's tokens as the prompt";
    sides
        .agree(
            line,
            UNKNOWN_MODEL_ID,
            RoutingPool::HttpRegular,
            None,
            &hinted,
        )
        .placed_on(line, &["http://m:1", "http://n:1"]);
}
