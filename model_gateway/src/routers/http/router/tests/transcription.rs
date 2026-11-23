//! The multipart transcription route over a pool that holds workers of
//! data-parallel ranks.
//!
//! A form names no rank to its worker, so the route hands the placement the
//! model's HTTP pool whole: a worker that registered a rank takes a
//! transcription like a plain one, and the form goes to its engine's own
//! address with no rank.
//!
//! A pool is a list of regular HTTP workers, each with its engine, its state
//! and the rank and group size it registered with, if any. The route's own
//! selection is asked sixteen times the pool's size: a rotation walks every
//! worker it may take, and a pool over on every such worker, steered by a
//! draw among the fewest in flight, reaches each of them. So an answer is
//! the whole set a request may land on, by each worker's place in the pool,
//! or the route's refusal.
//!
//! Each pool is asked under the classic `round_robin` policy and again with
//! the policy engine started over the document the same settings mean: the
//! route holds one rule and does not ask who places.

use std::{collections::BTreeSet, sync::Mutex};

use axum::http::{header::RETRY_AFTER, Uri};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use smg_policy_controller::document::AppliedSpec;
use smg_worker::test_support::no_health_check;

use super::super::*;
use crate::{
    app_context::testing::app_context_for_tests,
    config::{types::PolicyConfig, RouterConfig},
    routers::{common::placement::tests::engine_on, grpc::backend_client::BasicWorkerBuilder},
    tenant::TenantKey,
};

const MODEL: &str = "m";
const ROUTE: &str = "/v1/audio/transcriptions";
/// A pool is asked this many times its size.
const ROUNDS: usize = 16;
/// The stand-in engine's answer to a transcription.
const ANSWER: &str = "{\"text\":\"a transcription\"}";

/// What the router holds of a worker.
#[derive(Clone, Copy, PartialEq, Eq)]
enum State {
    /// Ready and under its overload thresholds.
    Free,
    /// Ready and over its overload thresholds.
    Over,
    /// Not ready.
    Down,
}

/// One worker of a pool, as it registers: its engine, by number, its state,
/// and the rank and the group size of its registration, if it names a rank.
#[derive(Clone, Copy)]
struct Member {
    engine: usize,
    state: State,
    rank: Option<(usize, usize)>,
}

/// The worker of an engine that registered no rank.
const fn plain(engine: usize) -> Member {
    Member {
        engine,
        state: State::Free,
        rank: None,
    }
}

/// The worker of rank `rank` of an engine of `of` ranks.
const fn ranked(engine: usize, rank: usize, of: usize) -> Member {
    Member {
        engine,
        state: State::Free,
        rank: Some((rank, of)),
    }
}

impl Member {
    const fn state(self, state: State) -> Self {
        Self { state, ..self }
    }
}

/// Who places a request: the classic policy, or the policy engine.
#[derive(Clone, Copy, Debug)]
enum Placer {
    Classic,
    Engine,
}

const PLACERS: [Placer; 2] = [Placer::Classic, Placer::Engine];

/// What the route's selection answers a pool.
#[derive(Debug, PartialEq, Eq)]
enum Answer {
    /// The places of the workers a request was placed on over the rounds.
    Placed(Vec<usize>),
    /// The refusal: its status, its code, and whether it says when to come
    /// back.
    Refused(u16, String, bool),
}

fn placed(places: &[usize]) -> Answer {
    Answer::Placed(places.to_vec())
}

fn refused(status: u16, code: &str, retry_after: bool) -> Answer {
    Answer::Refused(status, code.to_string(), retry_after)
}

/// A fresh pool in its members' state, under a fresh policy.
struct Bed {
    router: Router,
    workers: Vec<Arc<dyn Worker>>,
}

impl Bed {
    /// The pool of `members`, each engine at an address of its own that
    /// nothing listens on, with or without the shedding of a pool over on
    /// every worker (`--worker-overload-shed`).
    async fn of(members: &[Member], shed: bool, placer: Placer) -> Self {
        let address = |engine: usize| format!("http://127.0.0.1:{}", 9500 + engine);
        Self::at(address, members, shed, placer).await
    }

    /// The pool of `members`, each engine at the address `address` gives its
    /// number.
    async fn at(
        address: impl Fn(usize) -> String,
        members: &[Member],
        shed: bool,
        placer: Placer,
    ) -> Self {
        // The settings of a gateway under `round_robin`, and what its start
        // makes of them: the registry's shedding, and the document the
        // policy engine is started over.
        let config = RouterConfig {
            policy: PolicyConfig::RoundRobin,
            worker_overload_shed: shed,
            ..RouterConfig::default()
        };
        let document = config
            .policy_document()
            .expect("the settings mean a document");
        let worker_registry = Arc::new(WorkerRegistry::new());
        worker_registry.set_overload_shed(config.worker_overload_shed);
        let mut workers = Vec::new();
        for member in members {
            let mut builder = BasicWorkerBuilder::new(address(member.engine))
                .model(ModelCard::new(MODEL))
                .worker_type(WorkerType::Regular)
                .connection_mode(ConnectionMode::Http)
                .health_config(no_health_check());
            if let Some((rank, of)) = member.rank {
                builder = builder.dp_config(rank, of);
            }
            let worker: Arc<dyn Worker> = Arc::new(builder.build());
            worker.set_status(WorkerStatus::Ready);
            worker_registry
                .register(Arc::clone(&worker))
                .expect("a worker of the pool registers");
            worker.set_status(if member.state == State::Down {
                WorkerStatus::NotReady
            } else {
                WorkerStatus::Ready
            });
            worker_registry.set_worker_overloaded(&worker, member.state == State::Over);
            workers.push(worker);
        }
        // The router is built as the gateway builds it, by its own
        // constructor over a context: a field the router gains is filled
        // there, and this file names none of them.
        let context = app_context_for_tests(config, worker_registry, None);
        let router = Router::new(&context)
            .await
            .expect("the router builds over a test context");
        if matches!(placer, Placer::Engine) {
            let spec = AppliedSpec::validate(&document).expect("a valid document");
            engine_on(&router.policy_registry, spec);
        }
        Self { router, workers }
    }

    /// The route's answer to a transcription of the pool's model: the
    /// workers its selection placed the request on over the rounds, or its
    /// refusal.
    fn answer(&self) -> Answer {
        let routing = RoutingDerivatives {
            tokens: None,
            text: None,
            rid_key: None,
            cache_namespace: None,
        };
        let mut places = BTreeSet::new();
        for _ in 0..ROUNDS * self.workers.len().max(1) {
            match self
                .router
                .select_transcription_worker(None, MODEL, &routing, None)
            {
                Ok(worker) => {
                    let at = self
                        .workers
                        .iter()
                        .position(|w| w.url() == worker.url())
                        .expect("a worker of the pool");
                    places.insert(at);
                }
                Err(refusal) => {
                    assert!(places.is_empty(), "a request placed once is placed again");
                    return Answer::Refused(
                        refusal.status().as_u16(),
                        extract_error_code_from_response(&refusal).to_string(),
                        refusal.headers().contains_key(RETRY_AFTER),
                    );
                }
            }
        }
        Answer::Placed(places.into_iter().collect())
    }
}

/// Asks the pool of `members` under each placer in turn, and holds each
/// answer to `expected`.
async fn expect(members: &[Member], shed: bool, expected: &Answer) {
    for placer in PLACERS {
        let answer = Bed::of(members, shed, placer).await.answer();
        assert_eq!(&answer, expected, "{placer:?}, shedding {shed}");
    }
}

/// A model whose HTTP workers are the ranks of one engine is served: the
/// selection places a transcription, and each rank's worker takes its turn.
/// So is a model whose one worker is a group of one. No such pool is
/// refused for its ranks.
#[tokio::test]
async fn a_pool_of_rank_workers_alone_takes_a_transcription() {
    let two = [ranked(0, 0, 2), ranked(0, 1, 2)];
    expect(&two, false, &placed(&[0, 1])).await;
    let one = [ranked(0, 0, 1)];
    expect(&one, false, &placed(&[0])).await;
}

/// Where plain workers and rank workers share a model, every one of them
/// takes transcriptions: a rank's worker is the same send as a plain one.
#[tokio::test]
async fn a_rank_worker_beside_plain_ones_takes_its_share() {
    let mixed = [plain(0), ranked(1, 0, 2), ranked(1, 1, 2), plain(2)];
    expect(&mixed, false, &placed(&[0, 1, 2, 3])).await;
}

/// A plain worker that is not ready beside free rank workers: the rank
/// workers take the request, and the pool is not refused.
#[tokio::test]
async fn a_plain_worker_that_is_down_leaves_the_request_to_the_rank_workers() {
    let pool = [
        plain(0).state(State::Down),
        ranked(1, 0, 2),
        ranked(1, 1, 2),
    ];
    expect(&pool, false, &placed(&[1, 2])).await;
}

/// A plain worker over its thresholds yields to a free rank's worker,
/// whether a pool over on every worker is shed or not: this pool is not
/// over on every worker, so it is neither shed nor placed on the worker
/// that is over.
#[tokio::test]
async fn a_plain_worker_over_its_thresholds_yields_to_a_free_rank_worker() {
    let pool = [plain(0).state(State::Over), ranked(1, 0, 1)];
    for shed in [true, false] {
        expect(&pool, shed, &placed(&[1])).await;
    }
}

/// The refusals that stay hold over rank workers alone. No worker ready is
/// 503 `no_available_workers`. Every worker over its thresholds with
/// shedding on is 503 `worker_overload_protection_shed`. Each says when to
/// come back. A model no worker serves is 404 `model_not_found`. With
/// shedding off a pool over on every worker still takes the request, and a
/// rank's worker that is down leaves it to the other rank's.
#[tokio::test]
async fn the_refusals_that_stay_hold_over_rank_workers_alone() {
    let down = [
        ranked(0, 0, 2).state(State::Down),
        ranked(0, 1, 2).state(State::Down),
    ];
    expect(&down, false, &refused(503, "no_available_workers", true)).await;

    let over = [
        ranked(0, 0, 2).state(State::Over),
        ranked(0, 1, 2).state(State::Over),
    ];
    let shed = refused(503, "worker_overload_protection_shed", true);
    expect(&over, true, &shed).await;
    expect(&over, false, &placed(&[0, 1])).await;

    expect(&[], false, &refused(404, "model_not_found", false)).await;

    let half = [ranked(0, 0, 2).state(State::Down), ranked(0, 1, 2)];
    expect(&half, false, &placed(&[1])).await;
}

/// What the stand-in engine was sent: each request's path and body.
type Sent = Arc<Mutex<Vec<(String, Bytes)>>>;

/// A stand-in engine that answers a transcription with [`ANSWER`] and keeps
/// what it was sent. Returns its address.
#[expect(
    clippy::disallowed_methods,
    reason = "test stub server lives for the duration of the test process"
)]
async fn spawn_engine() -> (String, Sent) {
    let sent = Sent::default();
    let kept = Arc::clone(&sent);
    let transcribe = move |uri: Uri, body: Bytes| {
        let kept = Arc::clone(&kept);
        async move {
            kept.lock()
                .expect("the engine's requests")
                .push((uri.path().to_string(), body));
            ([(CONTENT_TYPE, "application/json")], ANSWER)
        }
    };
    let app = axum::Router::new().route(ROUTE, axum::routing::post(transcribe));
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
        .await
        .expect("a loopback port");
    let addr = listener.local_addr().expect("the port's address");
    tokio::spawn(async move {
        axum::serve(listener, app)
            .await
            .expect("the stand-in engine serves");
    });
    (format!("http://{addr}"), sent)
}

/// A transcription placed on a rank's worker is sent to its engine's own
/// address, as a form that names no rank, and the engine's answer is
/// relayed. Over a few requests each rank's worker is the routed one.
#[tokio::test]
async fn a_transcription_on_a_rank_worker_reaches_its_engine_with_no_rank() {
    const REQUESTS: usize = 4;
    let (address, sent) = spawn_engine().await;
    let ranks = [ranked(0, 0, 2), ranked(0, 1, 2)];
    let tenant = TenantRequestMeta::new(TenantKey::new("test-tenant"));
    for placer in PLACERS {
        let bed = Bed::at(|_| address.clone(), &ranks, false, placer).await;
        let mut routed = BTreeSet::new();
        for _ in 0..REQUESTS {
            let request = TranscriptionRequest {
                model: MODEL.to_string(),
                ..Default::default()
            };
            let audio = AudioFile {
                bytes: Bytes::from_static(b"not real audio"),
                file_name: "test.wav".to_string(),
                content_type: Some("audio/wav".to_string()),
            };
            let response = bed
                .router
                .route_audio_transcriptions(None, &tenant, &request, audio, MODEL)
                .await;
            assert_eq!(response.status(), StatusCode::OK, "{placer:?}");
            let worker = response
                .headers()
                .get("x-smg-routed-worker-id")
                .and_then(|id| id.to_str().ok())
                .expect("the routed worker's id")
                .to_string();
            routed.insert(worker);
            let body = to_bytes(response.into_body(), usize::MAX)
                .await
                .expect("the answer's body");
            assert_eq!(body.as_ref(), ANSWER.as_bytes(), "{placer:?}");
        }
        let workers: BTreeSet<String> = bed.workers.iter().map(|w| w.url().to_string()).collect();
        assert_eq!(routed, workers, "{placer:?}: each rank's worker is routed");
    }
    let sent = sent.lock().expect("the engine's requests");
    assert_eq!(sent.len(), REQUESTS * PLACERS.len());
    for (path, form) in sent.iter() {
        assert_eq!(path, ROUTE);
        let form = String::from_utf8_lossy(form);
        assert!(form.contains("name=\"file\""), "no audio part: {form}");
        assert!(form.contains("name=\"model\""), "no model part: {form}");
        assert!(
            !form.contains("data_parallel_rank"),
            "the form names a rank: {form}"
        );
    }
}
