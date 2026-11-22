//! Sticky sessions decided by the policy engine of this process, beside the
//! classic policies: one script of requests through a gateway each way.
//!
//! A case starts two gateways from one configuration, each as a start
//! builds it. With the setting off the registry's sticky selector places a
//! keyed request. With it on the policy engine does, by the session rule
//! the routing settings write on the policy's binding, and the registry
//! holds no selector. Both stand over the same loopback workers. The case
//! puts one script through each and reads which worker took each request:
//! a session lands where the classic policies put it, and moves as theirs
//! does when its worker leaves, drains, is overloaded or holds two of its
//! requests.

use std::{error::Error, sync::Arc, time::Duration};

use axum::{
    http::{header::CONTENT_TYPE, HeaderMap, HeaderValue, StatusCode},
    routing::post,
    Json,
};
use openai_protocol::{model_card::ModelCard, worker::WorkerStatus};
use serde_json::{json, Value};
use smg_worker::identity::basic::Worker;
use tokio::{
    net::TcpListener,
    sync::{mpsc, watch},
    task::JoinHandle,
    time::timeout,
};

use crate::{
    app_context::AppContext,
    config::{ManualAssignmentMode, PolicyConfig, RouterConfig, RoutingKeyOverrideConfig},
    middleware::{RouteRequestMeta, TenantKey},
    routers::{gateway::Gateway, grpc::backend_client::BasicWorkerBuilder, RouterTrait},
    server::ServerConfig,
};

type Case<T = ()> = Result<T, Box<dyn Error>>;

const MODEL: &str = "m";
/// The header a gateway reads a routing key from, at the setting's default.
const KEY_HEADER: &str = "x-smg-routing-key";
/// The prompt of a request its worker holds unanswered.
const HELD: &str = "held";
/// How long a step waits for a request to reach a worker, or for an answer.
const WAIT: Duration = Duration::from_secs(10);

/// Loopback workers that answer `/generate`. Each says which of them a
/// request reached, as it arrives, and holds the answer of a request whose
/// prompt is [`HELD`] until the fleet lets the held ones go.
struct Fleet {
    urls: Vec<String>,
    arrivals: mpsc::UnboundedReceiver<usize>,
    release: watch::Sender<bool>,
}

impl Fleet {
    #[expect(
        clippy::disallowed_methods,
        reason = "the test servers stop with their runtime"
    )]
    async fn of(workers: usize) -> Case<Self> {
        let (arrived, arrivals) = mpsc::unbounded_channel();
        let (release, _) = watch::channel(false);
        let mut urls = Vec::new();
        for at in 0..workers {
            let listener = TcpListener::bind("127.0.0.1:0").await?;
            urls.push(format!("http://{}", listener.local_addr()?));
            let arrived = arrived.clone();
            let release = release.clone();
            let answer = post(move |Json(body): Json<Value>| {
                let arrived = arrived.clone();
                let mut released = release.subscribe();
                async move {
                    // The case reads the channel to its end.
                    let _ = arrived.send(at);
                    if body["text"] == HELD {
                        // A fleet that is gone lets its requests go too.
                        let _ = released.wait_for(|released| *released).await;
                    }
                    ([(CONTENT_TYPE, "application/json")], "{}")
                }
            });
            let app = axum::Router::new().route("/generate", answer);
            tokio::spawn(async move { axum::serve(listener, app).await });
        }
        Ok(Self {
            urls,
            arrivals,
            release,
        })
    }

    fn url(&self, at: usize) -> Case<&str> {
        let url = self.urls.get(at).ok_or("no such worker")?;
        Ok(url.as_str())
    }

    /// The worker the next request reached.
    async fn arrived(&mut self) -> Case<usize> {
        Ok(timeout(WAIT, self.arrivals.recv())
            .await?
            .ok_or("the fleet is gone")?)
    }

    /// Lets every held request go, and reads each answer.
    async fn let_go(&self, held: &mut Vec<JoinHandle<StatusCode>>) -> Case {
        self.release.send_replace(true);
        for answer in held.drain(..) {
            assert_eq!(timeout(WAIT, answer).await??, StatusCode::OK);
        }
        self.release.send_replace(false);
        Ok(())
    }
}

/// One gateway over the fleet, as a start builds it.
struct Bed {
    context: Arc<AppContext>,
    gateway: Arc<Gateway>,
}

impl Bed {
    async fn start(config: RouterConfig, fleet: &Fleet) -> Case<Self> {
        let context = Box::pin(AppContext::from_config(config.clone(), 30, None, None)).await?;
        let context = Arc::new(context);
        let gateway = Box::pin(Gateway::from_config(
            &ServerConfig::minimal(config),
            &context,
        ))
        .await?;
        let bed = Self { context, gateway };
        for at in 0..fleet.urls.len() {
            bed.join(fleet, at)?;
        }
        Ok(bed)
    }

    /// Registers the fleet's worker `at`, ready, under a new worker object.
    fn join(&self, fleet: &Fleet, at: usize) -> Case {
        let worker = BasicWorkerBuilder::new(fleet.url(at)?)
            .model(ModelCard::new(MODEL))
            .status(WorkerStatus::Ready)
            .build();
        self.context
            .worker_registry
            .register(Arc::new(worker))
            .ok_or("the worker did not register")?;
        Ok(())
    }

    /// The registry's object of the fleet's worker `at`.
    fn worker(&self, fleet: &Fleet, at: usize) -> Case<Arc<dyn Worker>> {
        Ok(self
            .context
            .worker_registry
            .get_by_url(fleet.url(at)?)
            .ok_or("the worker is not registered")?)
    }
}

/// Where a request carries a key: its routing key header, and the `rid` of
/// its body.
#[derive(Clone, Copy, Debug)]
struct Keyed {
    header: Option<&'static str>,
    rid: Option<&'static str>,
}

const NO_KEY: Keyed = Keyed {
    header: None,
    rid: None,
};

const fn header(key: &'static str) -> Keyed {
    Keyed {
        header: Some(key),
        rid: None,
    }
}

const fn rid(rid: &'static str) -> Keyed {
    Keyed {
        header: None,
        rid: Some(rid),
    }
}

/// One generate request through `gateway`, and its status. The answer is
/// dropped here, so the request has left its worker when this returns.
async fn send(gateway: Arc<Gateway>, keyed: Keyed, prompt: &'static str) -> StatusCode {
    let meta = RouteRequestMeta::new(TenantKey::from("test-tenant"));
    let mut headers = HeaderMap::new();
    if let Some(key) = keyed.header {
        headers.insert(KEY_HEADER, HeaderValue::from_static(key));
    }
    let mut body = json!({ "text": prompt });
    if let Some(rid) = keyed.rid {
        body["rid"] = json!(rid);
    }
    let request = serde_json::from_value(body).expect("a generate request");
    gateway
        .route_generate(Some(&headers), &meta, request, MODEL)
        .await
        .status()
}

/// A worker of the fleet: the one that took the script's request of this
/// number, from 0.
#[derive(Clone, Copy, Debug)]
struct Took(usize);

/// One line of a script.
#[derive(Clone, Copy, Debug)]
enum Step {
    /// A request, answered before the next line.
    Ask(Keyed),
    /// A request its worker holds unanswered to the next [`Step::LetGo`].
    Hold(Keyed),
    /// The workers answer every request they hold.
    LetGo,
    /// The worker leaves the registry.
    Leave(Took),
    /// The worker registers again at its address.
    Join(Took),
    /// The worker's status moves: draining, not ready, ready again.
    Status(Took, WorkerStatus),
    /// The worker is overloaded, or no longer.
    Overloaded(Took, bool),
    /// The worker's object counts this many requests of the key in flight:
    /// what the load guard of each dispatch wrote.
    InFlight(Took, &'static str, usize),
}

use Step::{Ask, Hold, InFlight, Join, Leave, LetGo, Overloaded, Status};

/// The place in the fleet of the worker `who` names.
fn named(took: &[usize], who: Took) -> Case<usize> {
    Ok(*took.get(who.0).ok_or("no such request yet")?)
}

/// Puts `script` through a gateway of `config` over the fleet, and answers
/// with the worker that took each request, in the script's order. The
/// gateway decides as its setting says: with it on, the context holds the
/// engine object the routers ask and the registry no sticky selector.
#[expect(
    clippy::disallowed_methods,
    reason = "a held request waits in its own task, which the script joins"
)]
async fn run(config: RouterConfig, fleet: &mut Fleet, script: &[Step]) -> Case<Vec<usize>> {
    let by_engine = config.policy_controller;
    let bed = Bed::start(config, fleet).await?;
    assert_eq!(bed.context.policy_engine.is_some(), by_engine);
    assert_eq!(
        bed.context.policy_registry.holds_sticky_selector(),
        !by_engine
    );

    let mut took = Vec::new();
    let mut held = Vec::new();
    for (line, step) in script.iter().enumerate() {
        match *step {
            Ask(keyed) => {
                let status = send(Arc::clone(&bed.gateway), keyed, "hi").await;
                assert_eq!(status, StatusCode::OK, "line {line}");
                took.push(fleet.arrived().await?);
            }
            Hold(keyed) => {
                held.push(tokio::spawn(send(Arc::clone(&bed.gateway), keyed, HELD)));
                took.push(fleet.arrived().await?);
            }
            LetGo => fleet.let_go(&mut held).await?,
            Leave(who) => {
                let url = fleet.url(named(&took, who)?)?;
                bed.context
                    .worker_registry
                    .remove_by_url(url)
                    .ok_or("the worker was not registered")?;
            }
            Join(who) => bed.join(fleet, named(&took, who)?)?,
            Status(who, status) => bed.worker(fleet, named(&took, who)?)?.set_status(status),
            Overloaded(who, overloaded) => {
                let worker = bed.worker(fleet, named(&took, who)?)?;
                let moved = bed
                    .context
                    .worker_registry
                    .set_worker_overloaded(&worker, overloaded);
                assert!(moved, "line {line}");
            }
            InFlight(who, key, count) => {
                let worker = bed.worker(fleet, named(&took, who)?)?;
                assert_eq!(worker.routing_key_inflight(key), count, "line {line}");
            }
        }
        // One request reaches one worker, once.
        assert!(fleet.arrivals.try_recv().is_err(), "line {line}");
    }
    fleet.let_go(&mut held).await?;
    Ok(took)
}

/// `script` through a gateway with the classic policies, then through one
/// with the policy controller on: the worker that took each request, each
/// way.
async fn both(config: &RouterConfig, workers: usize, script: &[Step]) -> Case<[Vec<usize>; 2]> {
    let mut fleet = Fleet::of(workers).await?;
    let decided_by = |policy_controller| RouterConfig {
        policy_controller,
        ..config.clone()
    };
    let classic = Box::pin(run(decided_by(false), &mut fleet, script)).await?;
    let engine = Box::pin(run(decided_by(true), &mut fleet, script)).await?;
    Ok([classic, engine])
}

/// The workers of a list by the order they first appear in it: two lists
/// of one shape moved their requests alike, whichever worker a draw named
/// first.
fn shape(took: &[usize]) -> Vec<usize> {
    let mut seen = Vec::new();
    took.iter()
        .map(|worker| {
            seen.iter()
                .position(|known| known == worker)
                .unwrap_or_else(|| {
                    seen.push(*worker);
                    seen.len() - 1
                })
        })
        .collect()
}

/// A regular gateway of `policy` with sticky sessions on, a key's first
/// request assigned by `assignment_mode`, and no health probe behind its
/// case.
fn sticky(policy: PolicyConfig, assignment_mode: ManualAssignmentMode) -> RouterConfig {
    let mut config = RouterConfig {
        policy,
        routing_key_override: RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode,
            ..RoutingKeyOverrideConfig::default()
        },
        ..RouterConfig::default()
    };
    config.health_check.disable_health_check = true;
    config
}

/// The turns of one session land on one worker, each way. A session is the
/// routing key header without its turn and retry suffixes, or the
/// request's own id without them where no header holds a key, and the
/// header is read first. A request with no key, and each session's first
/// request under `delegate`, is placed by the policy: round robin's
/// rotation gives each of them the next worker, from the worker it starts
/// at, which each decider draws for itself.
#[tokio::test]
async fn a_session_lands_on_the_worker_the_classic_policies_put_it_on() -> Case {
    let both_sources = Keyed {
        header: Some("conv_t3"),
        rid: Some("talk_t3"),
    };
    let script = [
        Ask(NO_KEY),
        Ask(header("conv_t1")),
        Ask(NO_KEY),
        Ask(header("conv_t2")),
        Ask(header("conv_t2_r1")),
        Ask(rid("talk_t1")),
        Ask(NO_KEY),
        Ask(rid("talk_t2_r1")),
        Ask(both_sources),
        Ask(rid("talk_t3")),
        Ask(header("other")),
    ];
    let config = sticky(PolicyConfig::RoundRobin, ManualAssignmentMode::Delegate);
    let [classic, engine] = both(&config, 3, &script).await?;
    let engine = shape(&engine);
    assert_eq!(engine, shape(&classic));
    // The rotation: the keyless requests and each session's first.
    assert_eq!([0, 1, 2, 5, 6, 10].map(|n| engine[n]), [0, 1, 2, 0, 1, 2]);
    // `conv` by its header, with the id of another session beside it once.
    assert_eq!([3, 4, 8].map(|n| engine[n]), [engine[1]; 3]);
    // `talk` by the request's own id.
    assert_eq!([7, 9].map(|n| engine[n]), [engine[5]; 2]);
    Ok(())
}

/// A session whose worker stops taking requests moves to another worker,
/// and what it does when the first takes requests again is the classic
/// rule's, by the assignment: under `delegate` the worker placed last
/// leads the session's pins and keeps it, and under the sticky map's own
/// assignments the first leads again and takes it back. The worker stops
/// in each way a worker does: it leaves the registry and registers again
/// at its address, it drains, it fails its probes, it is overloaded.
#[tokio::test]
async fn a_session_whose_worker_stops_moves_as_under_the_classic_policies() -> Case {
    let first = Took(0);
    let ways = [
        ("leaves", Leave(first), Join(first)),
        (
            "drains",
            Status(first, WorkerStatus::Draining),
            Status(first, WorkerStatus::Ready),
        ),
        (
            "is not ready",
            Status(first, WorkerStatus::NotReady),
            Status(first, WorkerStatus::Ready),
        ),
        (
            "is overloaded",
            Overloaded(first, true),
            Overloaded(first, false),
        ),
    ];
    for (assignment_mode, back_on) in [
        (ManualAssignmentMode::Delegate, 1),
        (ManualAssignmentMode::MinLoad, 0),
    ] {
        for (way, stops, returns) in ways {
            let turn = Ask(header("conv_t1"));
            let script = [turn, turn, stops, turn, turn, returns, turn, turn];
            let config = sticky(PolicyConfig::RoundRobin, assignment_mode);
            let [classic, engine] = both(&config, 3, &script).await?;
            let said = format!("its worker {way}, under {assignment_mode:?}");
            assert_eq!(shape(&engine), shape(&classic), "{said}");
            assert_eq!(shape(&engine), [0, 0, 1, 1, back_on, back_on], "{said}");
        }
    }
    Ok(())
}

/// A worker carries two requests of one session. The third is placed again,
/// by the policy under `delegate`, and the session's pin moves with it to
/// a worker under that count: the turns behind it follow. The count is the
/// load guard's, kept on the worker object under the session's key: the
/// header's key and the id's alike, each without its suffixes.
#[tokio::test]
async fn a_session_at_its_cap_on_a_worker_is_placed_again() -> Case {
    for (turns, key) in [
        (
            ["conv_t1", "conv_t2", "conv_t3", "conv_t4_r1", "conv_t5"].map(header),
            "conv",
        ),
        (
            ["talk_t1", "talk_t2", "talk_t3", "talk_t4_r1", "talk_t5"].map(rid),
            "talk",
        ),
    ] {
        for assignment_mode in [
            ManualAssignmentMode::Delegate,
            ManualAssignmentMode::MinLoad,
        ] {
            let [first, second, third, fourth, fifth] = turns;
            let pinned = Took(0);
            let script = [
                Ask(first),
                Hold(second),
                InFlight(pinned, key, 1),
                Hold(third),
                InFlight(pinned, key, 2),
                Ask(fourth),
                InFlight(pinned, key, 2),
                LetGo,
                InFlight(pinned, key, 0),
                Ask(fifth),
                Ask(fifth),
            ];
            let config = sticky(PolicyConfig::RoundRobin, assignment_mode);
            let [classic, engine] = both(&config, 3, &script).await?;
            let said = format!("{key} under {assignment_mode:?}");
            assert_eq!(shape(&engine), shape(&classic), "{said}");
            assert_eq!(shape(&engine), [0, 0, 0, 1, 1, 1], "{said}");
        }
    }
    Ok(())
}

/// Where the policy picks the capped worker again, the request goes to it
/// and the pin stays: the pin moves only to another worker under the cap.
/// Round robin over two workers names the pinned one at the session's
/// first request past the cap and the other at the next, each way.
#[tokio::test]
async fn a_capped_session_keeps_its_pin_where_the_policy_picks_its_worker_again() -> Case {
    let script = [
        Ask(header("conv_t1")),
        Ask(NO_KEY),
        Hold(header("conv_t2")),
        Hold(header("conv_t3")),
        Ask(header("conv_t4")),
        Ask(header("conv_t5")),
        LetGo,
        Ask(header("conv_t6")),
    ];
    let config = sticky(PolicyConfig::RoundRobin, ManualAssignmentMode::Delegate);
    let [classic, engine] = both(&config, 2, &script).await?;
    assert_eq!(shape(&engine), shape(&classic));
    assert_eq!(shape(&engine), [0, 1, 0, 0, 0, 1, 1]);
    Ok(())
}

/// Under `manual` the policy reads the routing key itself, and the override
/// adds the request's own id to its sources and takes the turn and retry
/// suffixes off each. A session by its id keeps its worker each way, and
/// so does one by its header: with the setting on the manual policy's own
/// session rule holds both, and the registry holds no sticky map beside
/// it. The first session's request is held on its worker, so the fewest
/// requests in flight put the second session on another.
#[tokio::test]
async fn under_manual_the_override_adds_the_requests_id_each_way() -> Case {
    let manual = PolicyConfig::Manual {
        eviction_interval_secs: 60,
        max_idle_secs: 14_400,
        assignment_mode: ManualAssignmentMode::MinLoad,
    };
    let script = [
        Hold(rid("talk_t1")),
        Ask(header("conv_t1")),
        Ask(rid("talk_t2_r1")),
        Ask(header("conv_t2_r3")),
        LetGo,
        Ask(rid("talk_t3")),
        Ask(header("conv_t3")),
    ];
    let config = sticky(manual, ManualAssignmentMode::Delegate);
    let [classic, engine] = both(&config, 3, &script).await?;
    assert_eq!(shape(&engine), shape(&classic));
    assert_eq!(shape(&engine), [0, 1, 0, 1, 0, 1]);
    Ok(())
}
