//! A gateway's decide stream: which worker a present gateway is answered, a
//! worker added over the wire, a frame held to the workers that meet its
//! stated demand, and a dispatch that feeds a registered scorer.

use std::{
    io::Write,
    time::{Duration, Instant},
};

use smg_grand_central::{
    document::{AppliedSpec, PolicySpec},
    engine::{fingerprint, Role, WorkerId},
    service::{
        controller::{Config, Controller, Scope, State},
        proto::{
            self, busy, controller_frame, decide_answer, decision, gateway_frame,
            grand_central_client::GrandCentralClient, Credit, DecideRequest, Decision,
            GatewayFrame, Hello, LeaderRequest, RequestFacts,
        },
    },
    store::{GatewayId, LeaseState, LeasesStore, WorkerFacts},
    tracking::LeaseId,
};
use tokio::{
    sync::{mpsc, oneshot},
    time::timeout,
};
use tonic::{codegen::tokio_stream::wrappers::ReceiverStream, Code};

use crate::{config, net::http_get, ready, stating};

/// A gateway's whole way to "which worker": no decide stream before its
/// `Hello`, the `Welcome`'s window and cap, then a window's burst of
/// requests on one decide stream, each answered with a registered worker
/// under the leader's term, and the credit the answers give back. A
/// request the controller answers `Busy` is on its scrape, at its reason.
#[tokio::test]
async fn a_present_gateway_is_answered_which_worker_on_its_decide_stream() {
    const GATEWAY: &str = "gateway-a";
    const MODEL: &str = "model-a";
    let mut controller = Controller::bind(config()).await.expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let (store, holder) = (controller.store(), controller.workers());
    // Three ready members serving the model, under a preset of this build.
    let mut workers = Vec::new();
    for name in ["w0", "w1", "w2"] {
        let key = format!("http://{name}.fleet:8000");
        let stated = WorkerFacts {
            key: key.clone(),
            engine: key,
            incarnation: 1,
            role: Role::Single,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            max_running: Some(256),
            ..WorkerFacts::default()
        };
        let id = ready(&holder, &store, stated).expect("a ready worker");
        workers.push(id.0);
    }
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(spec).expect("above the unbound start");

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let addr = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let mut client = GrandCentralClient::connect(format!("http://{addr}"))
            .await
            .expect("the listener accepts");

        // Before its Hello the gateway is not present.
        let (_unsent, sent) = mpsc::channel::<DecideRequest>(1);
        let refused = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect_err("no session of the gateway stands");
        assert_eq!(refused.code(), Code::FailedPrecondition);
        assert_eq!(
            refused.message(),
            "no session of this gateway stands on this controller: a decide stream is a present gateway's (say Hello on a Session stream first)"
        );

        let (frames, sent) = mpsc::channel(4);
        let mut session = client
            .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("the stream opens")
            .into_inner();
        let hello = Hello {
            gateway_id: GATEWAY.to_owned(),
            protocol_min: 1,
            protocol_max: 1,
            ..Hello::default()
        };
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor: 0,
                frame: Some(gateway_frame::Frame::Hello(hello)),
            })
            .await
            .expect("the stream takes the Hello");
        let answer = timeout(Duration::from_secs(20), session.message())
            .await
            .expect("a Welcome within the case's patience")
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        let window = match answer.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => {
                assert_eq!(welcome.max_decide_streams, 256);
                welcome.decide_window
            }
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };
        assert_eq!(window, 128);

        // The whole window at once: the controller's queues hold a
        // stream's window, so none comes back for their depth.
        let (requests, sent) = mpsc::channel(256);
        let mut answers = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a present gateway's stream opens")
            .into_inner();
        for id in 0..window {
            requests
                .send(DecideRequest {
                    request_id: id.to_be_bytes().to_vec(),
                    facts: Some(RequestFacts {
                        model: MODEL.to_owned(),
                        seed: u64::from(id),
                        budget_micros: 5_000_000,
                        ..RequestFacts::default()
                    }),
                    constraints: None,
                    again: false,
                    ..Default::default()
                })
                .await
                .expect("the stream takes the request");
        }
        // The answers in the requests' order, a Credit frame of half the
        // window behind each half of them.
        let (mut decided, mut credit) = (0u32, 0u32);
        while credit < window {
            let answer = timeout(Duration::from_secs(20), answers.message())
                .await
                .expect("an answer within the case's patience")
                .expect("an answer, not a status")
                .expect("an answer, not the stream's end");
            assert_eq!(answer.term, 1, "the first take's term: {answer:?}");
            match answer.answer {
                Some(decide_answer::Answer::Decision(Decision {
                    outcome: Some(decision::Outcome::Pick(pick)),
                    ..
                })) => {
                    assert_eq!(answer.request_id, decided.to_be_bytes());
                    let target = pick.targets.first().expect("a target").id;
                    assert!(workers.contains(&target), "{target}");
                    decided += 1;
                }
                Some(decide_answer::Answer::Credit(Credit { window: grant })) => {
                    assert!(answer.request_id.is_empty());
                    assert_eq!((grant, decided % grant), (window / 2, 0));
                    credit += grant;
                }
                other => panic!("a pick or a credit: {other:?}"),
            }
        }
        assert_eq!(decided, window);

        // A request that brings no budget has none left: the gateway's to
        // decide, and counted at its reason on the controller's scrape.
        requests
            .send(DecideRequest {
                request_id: vec![0xff],
                facts: Some(RequestFacts {
                    model: MODEL.to_owned(),
                    ..RequestFacts::default()
                }),
                constraints: None,
                again: false,
                ..Default::default()
            })
            .await
            .expect("the stream takes the request");
        let answer = timeout(Duration::from_secs(20), answers.message())
            .await
            .expect("an answer within the case's patience")
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        match answer.answer {
            Some(decide_answer::Answer::Busy(told)) => {
                assert_eq!(told.reason(), busy::Reason::BudgetGoneOnArrival);
            }
            other => panic!("a Busy: {other:?}"),
        }
        let (status, scrape) = http_get(addresses.admin, "/metrics").await.expect("GET");
        assert_eq!(status, 200);
        for line in [
            "smg_grand_central_decide_busy_total{reason=\"budget_gone_on_arrival\"} 1",
            "smg_grand_central_decide_busy_total{reason=\"queue_depth\"} 0",
            "smg_grand_central_decide_no_decision_total{cause=\"fault\"} 0",
            "smg_grand_central_decide_answers_dropped_total{cause=\"queue_full\"} 0",
        ] {
            assert!(scrape.contains(line), "{line}\n{scrape}");
        }

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// A worker added through the control listener is decided for a present
/// gateway with no ask for a build anywhere: the served controller's own
/// loop pays the build the registration owes. Its tick alone pays inside
/// this case's bound, so the wake that pays at the window's end is held by
/// the loop's own case, under paused time.
#[tokio::test]
async fn a_worker_added_over_the_wire_is_decided_with_no_ask_for_a_build() {
    use smg_grand_central::service::proto::{
        register_worker_request::Mode, worker_event::Event, RegisterWorkerRequest, WireMode,
        WorkerSpec,
    };
    const GATEWAY: &str = "gateway-w";
    const MODEL: &str = "model-w";
    let mut controller = Controller::bind(config()).await.expect("bind");
    for name in ["gateway", "control"] {
        controller
            .identity_as_told(name)
            .expect("a listener of the configuration");
    }
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let store = controller.store();
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(spec).expect("above the unbound start");

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let gateway = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let control = addresses
            .of(Scope::Control)
            .expect("every scope is mounted");
        let mut client = GrandCentralClient::connect(format!("http://{gateway}"))
            .await
            .expect("the listener accepts");
        let mut operator = GrandCentralClient::connect(format!("http://{control}"))
            .await
            .expect("the listener accepts");
        let (frames, sent) = mpsc::channel(4);
        let mut session = client
            .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("the stream opens")
            .into_inner();
        let hello = Hello {
            gateway_id: GATEWAY.to_owned(),
            protocol_min: 1,
            protocol_max: 1,
            ..Hello::default()
        };
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor: 0,
                frame: Some(gateway_frame::Frame::Hello(hello)),
            })
            .await
            .expect("the stream takes the Hello");
        let answer = timeout(Duration::from_secs(20), session.message())
            .await
            .expect("a Welcome within the case's patience")
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        assert!(
            matches!(answer.frame, Some(controller_frame::Frame::Welcome(_))),
            "a Welcome answers a Hello: {answer:?}"
        );
        let (requests, sent) = mpsc::channel(4);
        let mut answers = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a present gateway's stream opens")
            .into_inner();

        // The add of a worker whose policy turns its probes off: the call's
        // connect signal readies it. No call asks for a build: the served
        // loop pays it.
        let builds = store.membership_builds();
        let request = RegisterWorkerRequest {
            spec: Some(WorkerSpec {
                url: "http://w.fleet:8000".to_owned(),
                wire: WireMode::Grpc as i32,
                role: proto::Role::Single as i32,
                models: vec![MODEL.to_owned()],
                health_policy: Some(proto::HealthPolicy {
                    disabled: true,
                    ..Default::default()
                }),
                ..WorkerSpec::default()
            }),
            mode: Mode::Add as i32,
            wait_ready: false,
        };
        let added = operator
            .register_worker(stating(Some("operator-w"), request))
            .await
            .expect("the add is taken")
            .into_inner();
        let id = match added.event {
            Some(Event::Registered(row)) => WorkerId(row.id),
            other => panic!("a registration's answer: {other:?}"),
        };
        let answered = Instant::now();

        // One request at a time, until a pick names the worker.
        let mut asked = 0u32;
        loop {
            asked += 1;
            requests
                .send(DecideRequest {
                    request_id: asked.to_be_bytes().to_vec(),
                    facts: Some(RequestFacts {
                        model: MODEL.to_owned(),
                        seed: u64::from(asked),
                        budget_micros: 5_000_000,
                        ..RequestFacts::default()
                    }),
                    constraints: None,
                    again: false,
                    ..Default::default()
                })
                .await
                .expect("the stream takes the request");
            let picked = loop {
                let answer = timeout(Duration::from_secs(20), answers.message())
                    .await
                    .expect("an answer within the case's patience")
                    .expect("an answer, not a status")
                    .expect("an answer, not the stream's end");
                match answer.answer {
                    Some(decide_answer::Answer::Credit(_)) => {}
                    Some(decide_answer::Answer::Decision(Decision {
                        outcome: Some(decision::Outcome::Pick(pick)),
                        ..
                    })) => break pick.targets.first().map(|target| target.id),
                    _ => break None,
                }
            };
            // The bound holds for the pick too: one that comes late is no
            // pass.
            assert!(
                answered.elapsed() < Config::DEFAULT_LOAD_POLL_INTERVAL / 4,
                "not decided {:?} after its answer, over {asked} requests",
                answered.elapsed()
            );
            if picked == Some(id.0) {
                break;
            }
        }
        assert_eq!(store.membership_builds(), builds + 1);

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// What a frame's answer names: every worker of its pick, or the stage
/// that shed its request with no endpoint.
type Named = Result<Vec<u64>, String>;

/// One frame for `model` that states `demands`, numbered `n`, and what its
/// answer names.
async fn demanded(
    requests: &mpsc::Sender<DecideRequest>,
    answers: &mut tonic::Streaming<proto::DecideAnswer>,
    n: u32,
    model: &str,
    demands: Option<proto::Demands>,
) -> Result<Named, String> {
    requests
        .send(DecideRequest {
            request_id: n.to_be_bytes().to_vec(),
            facts: Some(RequestFacts {
                model: model.to_owned(),
                seed: u64::from(n),
                budget_micros: 5_000_000,
                ..RequestFacts::default()
            }),
            demands,
            ..Default::default()
        })
        .await
        .map_err(|_| "the stream takes no request".to_owned())?;
    loop {
        let answer = timeout(Duration::from_secs(20), answers.message())
            .await
            .map_err(|_| "no answer within the case's patience".to_owned())?
            .map_err(|status| format!("a status, not an answer: {status:?}"))?
            .ok_or("the stream's end, not an answer")?;
        match answer.answer {
            Some(decide_answer::Answer::Credit(_)) => {}
            Some(decide_answer::Answer::Decision(Decision {
                outcome: Some(decision::Outcome::Pick(pick)),
                ..
            })) => return Ok(Ok(pick.targets.iter().map(|target| target.id).collect())),
            Some(decide_answer::Answer::Decision(Decision {
                outcome: Some(decision::Outcome::Shed(shed)),
                ..
            })) if shed.reason() == proto::ShedReason::NoEndpoints => {
                return Ok(Err(shed.emptied_by));
            }
            other => {
                return Err(format!(
                    "neither a pick nor a shed for no endpoint: {other:?}"
                ))
            }
        }
    }
}

/// A frame states its request's hard demands, and a controller holds the
/// request to the workers whose registrations meet them.
///
/// Three workers are added through the control listener. One states a
/// runtime, media by reference under its registering party's pipeline, a
/// window and a way to decide a constraint. One states another runtime, a
/// pipeline of its own and another way, and no window. One states nothing.
///
/// A gateway that speaks protocol 1 alone reads a `Leader` range of 1 to
/// 2 and is welcomed on 1. A demand its frame states is read all the same.
/// Every pick of a demand names workers that meet it and no other. A
/// demand no worker meets is shed with no endpoint, under that demand's
/// own stage. A frame at a side's phase that states a demand ends its
/// stream by the field.
#[tokio::test]
async fn a_frame_that_states_a_demand_is_held_to_the_workers_whose_registrations_meet_it() {
    use std::collections::BTreeSet;

    use smg_grand_central::service::proto::{
        demands::References, register_worker_request::Mode, worker_event::Event, DemandFacts,
        Demands, RegisterWorkerRequest, WireMode, WorkerSpec,
    };
    const GATEWAY: &str = "gateway-d";
    const MODEL: &str = "model-d";
    /// The asks a demand is given: each names workers that meet it, and
    /// together they name every such worker.
    const ASKS: u32 = 48;
    let mut controller = Controller::bind(config()).await.expect("bind");
    for name in ["gateway", "control"] {
        controller
            .identity_as_told(name)
            .expect("a listener of the configuration");
    }
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let store = controller.store();
    let mut document = PolicySpec::preset("random").expect("a preset of this build");
    document.generation = 1;
    let spec = AppliedSpec::validate(&document).expect("the preset validates");
    store.apply(spec).expect("above the unbound start");

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let gateway = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let control = addresses
            .of(Scope::Control)
            .expect("every scope is mounted");
        let mut client = GrandCentralClient::connect(format!("http://{gateway}"))
            .await
            .expect("the listener accepts");
        let mut operator = GrandCentralClient::connect(format!("http://{control}"))
            .await
            .expect("the listener accepts");
        // What a peer that speaks protocol 1 alone sees of this build: a
        // range that holds its number, and a Welcome on it.
        let leader = client
            .leader(LeaderRequest::default())
            .await
            .expect("Leader asks for no principal")
            .into_inner();
        assert_eq!((leader.protocol_min, leader.protocol_max), (1, 2));
        let (frames, sent) = mpsc::channel(4);
        let mut session = client
            .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("the stream opens")
            .into_inner();
        let hello = Hello {
            gateway_id: GATEWAY.to_owned(),
            protocol_min: 1,
            protocol_max: 1,
            ..Hello::default()
        };
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor: 0,
                frame: Some(gateway_frame::Frame::Hello(hello)),
            })
            .await
            .expect("the stream takes the Hello");
        let answer = timeout(Duration::from_secs(20), session.message())
            .await
            .expect("a Welcome within the case's patience")
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        let welcomed = match answer.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => Some(welcome.protocol),
            _ => None,
        };
        assert_eq!(
            welcomed,
            Some(1),
            "a peer that offers 1 alone is welcomed on 1"
        );
        let (requests, sent) = mpsc::channel(4);
        let mut answers = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a present gateway's stream opens")
            .into_inner();

        // Three workers whose policy turns their probes off: each call's
        // connect signal readies its worker.
        let stated = [
            Some(DemandFacts {
                runtime: 7,
                references: true,
                host_pipeline: true,
                window: 4_096,
                decoding: 3,
            }),
            Some(DemandFacts {
                runtime: 8,
                references: true,
                host_pipeline: false,
                window: 0,
                decoding: 4,
            }),
            None,
        ];
        let mut ids = Vec::with_capacity(stated.len());
        for (i, block) in stated.into_iter().enumerate() {
            let request = RegisterWorkerRequest {
                spec: Some(WorkerSpec {
                    url: format!("http://d{i}.fleet:8000"),
                    wire: WireMode::Grpc as i32,
                    role: proto::Role::Single as i32,
                    models: vec![MODEL.to_owned()],
                    health_policy: Some(proto::HealthPolicy {
                        disabled: true,
                        ..Default::default()
                    }),
                    demand_facts: block,
                    ..WorkerSpec::default()
                }),
                mode: Mode::Add as i32,
                wait_ready: false,
            };
            let added = operator
                .register_worker(stating(Some("operator-d"), request))
                .await
                .expect("the add is taken")
                .into_inner();
            match added.event {
                Some(Event::Registered(row)) => {
                    assert_eq!(row.demand_facts, block, "the answer tells the block");
                    ids.push(row.id);
                }
                other => panic!("a registration's answer: {other:?}"),
            }
        }
        let [first, second, third] = ids[..] else {
            panic!("three workers");
        };
        let all = BTreeSet::from([first, second, third]);

        // Frames that state no demand, until their picks have named the
        // three: the served loop pays the builds the registrations owe.
        let mut asked = 0u32;
        let mut named = BTreeSet::new();
        while named != all {
            asked += 1;
            assert!(
                asked < 4_096,
                "the three workers are not decided: {named:?}"
            );
            let answer = demanded(&requests, &mut answers, asked, MODEL, None).await;
            if let Ok(targets) = answer.expect("an answer") {
                named.extend(targets);
            }
        }

        let none = Demands::default();
        let at = |references: References| Demands {
            references: references.into(),
            ..none
        };
        let both = BTreeSet::from([first, second]);
        let wide = BTreeSet::from([second, third]);
        let picks = [
            (
                "a runtime",
                Demands { runtime: 7, ..none },
                BTreeSet::from([first]),
            ),
            (
                "the other runtime",
                Demands { runtime: 8, ..none },
                BTreeSet::from([second]),
            ),
            ("media by reference", at(References::Any), both.clone()),
            (
                "the party's pipeline",
                at(References::Caller),
                BTreeSet::from([first]),
            ),
            (
                "a pipeline of its own",
                at(References::Worker),
                BTreeSet::from([second]),
            ),
            (
                "an input its window holds",
                Demands {
                    input_window: 4_096,
                    ..none
                },
                all.clone(),
            ),
            (
                "an input past a window",
                Demands {
                    input_window: 4_097,
                    ..none
                },
                wide.clone(),
            ),
            (
                "an output past a window",
                Demands {
                    output_window: 4_097,
                    ..none
                },
                wide,
            ),
            (
                "a way to decide",
                Demands {
                    decoding: 3,
                    ..none
                },
                BTreeSet::from([first]),
            ),
            (
                "a runtime and media by reference",
                Demands {
                    runtime: 8,
                    ..at(References::Any)
                },
                BTreeSet::from([second]),
            ),
        ];
        for (what, demands, meeting) in picks {
            let mut named = BTreeSet::new();
            for _ in 0..ASKS {
                asked += 1;
                let answer = demanded(&requests, &mut answers, asked, MODEL, Some(demands)).await;
                let answer = answer.expect("an answer");
                named.extend(answer.unwrap_or_else(|stage| panic!("{what} is shed by {stage}")));
            }
            assert_eq!(named, meeting, "{what}");
        }
        let sheds = [
            (
                "a runtime none states",
                Demands { runtime: 9, ..none },
                "wires",
            ),
            (
                "a way none states",
                Demands {
                    decoding: 5,
                    ..none
                },
                "decoding",
            ),
            (
                "vision",
                Demands {
                    vision: true,
                    ..none
                },
                "vision",
            ),
        ];
        for (what, demands, stage) in sheds {
            asked += 1;
            let answer = demanded(&requests, &mut answers, asked, MODEL, Some(demands)).await;
            assert_eq!(answer, Ok(Err(stage.to_owned())), "{what}");
        }

        // A frame at a side's phase holds no demand yet: one that states a
        // demand is out of range by the field, and its stream ends.
        requests
            .send(DecideRequest {
                request_id: vec![0xd],
                facts: Some(RequestFacts {
                    model: MODEL.to_owned(),
                    phase: proto::Phase::Prefill.into(),
                    budget_micros: 5_000_000,
                    ..RequestFacts::default()
                }),
                demands: Some(Demands { runtime: 7, ..none }),
                ..Default::default()
            })
            .await
            .expect("the stream takes the request");
        let ended = loop {
            let next = timeout(Duration::from_secs(20), answers.message())
                .await
                .expect("the stream's end within the case's patience");
            match next {
                Ok(Some(frame))
                    if matches!(frame.answer, Some(decide_answer::Answer::Credit(_))) => {}
                Ok(other) => panic!("the stream ends by a status: {other:?}"),
                Err(status) => break status,
            }
        };
        assert_eq!(ended.code(), Code::InvalidArgument, "{ended:?}");
        assert_eq!(
            ended.message(),
            "demands is out of its range in this protocol version"
        );

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

#[tokio::test]
async fn a_served_dispatch_feeds_a_registered_scorer_without_lifecycle_hooks() {
    use std::sync::{Arc, Mutex};

    use smg_grand_central::{
        document::API_VERSION,
        engine::{Candidate, Scorer, TieBand},
        service::proto::{Dispatched, Side},
        stages::{Lifecycle, StageRegistry},
        tracking::{CreditedDispatches, DispatchCredit},
    };

    #[derive(Debug)]
    struct SentScore {
        sent: Arc<DispatchCredit>,
        seen: Arc<Mutex<Vec<(WorkerId, CreditedDispatches)>>>,
    }

    impl Lifecycle for SentScore {}

    impl Scorer for SentScore {
        fn name(&self) -> &'static str {
            "sent-score"
        }

        fn score(
            &self,
            facts: &smg_grand_central::engine::RequestFacts<'_>,
            candidates: &[Candidate],
            kept: &[u32],
            out: &mut [f32],
        ) -> TieBand {
            let reads = self.sent.pass(facts);
            let mut seen = self.seen.lock().expect("the scores");
            seen.clear();
            for (&index, score) in kept.iter().zip(out) {
                let row = &candidates[index as usize];
                let sent = reads.of(row);
                seen.push((row.id, sent));
                *score = if sent.requests == 0 { 1.0 } else { 0.0 };
            }
            TieBand::NORMALIZED
        }
    }

    const GATEWAY: &str = "sender";
    const MODEL: &str = "work";
    const TOKENS: u32 = 367;
    const PATIENCE: Duration = Duration::from_secs(12);

    let seen = Arc::new(Mutex::new(Vec::new()));
    let mut catalog = StageRegistry::builtin();
    let scores = Arc::clone(&seen);
    catalog
        .scorer_with("sent-score", move |build| {
            Ok(Box::new(SentScore {
                sent: build.dispatch_credit(),
                seen: Arc::clone(&scores),
            }))
        })
        .expect("a free name");
    let mut controller = Controller::bind_with(config(), catalog)
        .await
        .expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("the listener");

    let mut file = tempfile::NamedTempFile::new().expect("a policy file");
    write!(
        file,
        "apiVersion: {API_VERSION}\nprofiles:\n  work:\n    sent_ttl_ms: 24001\n    score: [sent-score]\n    pick: best\n"
    )
    .expect("the document is written");
    controller
        .spec_file(file.path())
        .await
        .expect("the document validates");

    let (store, holder) = (controller.store(), controller.workers());
    for key in ["http://first:8000", "http://second:8000"] {
        let stated = WorkerFacts {
            key: key.to_owned(),
            engine: key.to_owned(),
            incarnation: 1,
            models: vec![MODEL.to_owned()],
            dp_size: 1,
            ..WorkerFacts::default()
        };
        ready(&holder, &store, stated).expect("a ready worker");
    }
    store.publish_now();

    let mut readiness = controller.readiness();
    let addr = controller
        .addresses()
        .of(Scope::Gateway)
        .expect("the gateway listener");
    let (stop, stopped) = oneshot::channel::<()>();
    let (served, ()) = tokio::join!(
        controller.serve(async {
            let _ = stopped.await;
        }),
        async {
            timeout(PATIENCE, async {
                assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
                let mut client = GrandCentralClient::connect(format!("http://{addr}"))
                    .await
                    .expect("the listener accepts");
                let (frames, sent) = mpsc::channel(4);
                let mut session = client
                    .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
                    .await
                    .expect("a session")
                    .into_inner();
                frames
                    .send(GatewayFrame {
                        frame: Some(gateway_frame::Frame::Hello(Hello {
                            gateway_id: GATEWAY.to_owned(),
                            protocol_min: 1,
                            protocol_max: 1,
                            ..Hello::default()
                        })),
                        ..GatewayFrame::default()
                    })
                    .await
                    .expect("the Hello");
                let welcome = session
                    .message()
                    .await
                    .expect("a frame")
                    .expect("a Welcome");
                assert!(matches!(
                    welcome.frame,
                    Some(controller_frame::Frame::Welcome(_))
                ));

                let (requests, sent) = mpsc::channel(4);
                let mut answers = client
                    .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
                    .await
                    .expect("a decide stream")
                    .into_inner();
                let mut first = None;
                for id in [b"a", b"b"] {
                    requests
                        .send(DecideRequest {
                            request_id: id.to_vec(),
                            facts: Some(RequestFacts {
                                model: MODEL.to_owned(),
                                prompt_tokens: TOKENS,
                                budget_micros: 1_470_000,
                                ..RequestFacts::default()
                            }),
                            constraints: None,
                            again: false,
                            ..Default::default()
                        })
                        .await
                        .expect("a request");
                    let answer = answers
                        .message()
                        .await
                        .expect("an answer")
                        .expect("a decision");
                    let Some(decide_answer::Answer::Decision(answer)) = answer.answer else {
                        panic!("a decision answers the request: {answer:?}");
                    };
                    let Some(decision::Outcome::Pick(pick)) = answer.outcome else {
                        panic!("the decision picks a worker: {answer:?}");
                    };
                    let target = WorkerId(pick.targets[0].id);
                    if let Some(dispatched) = first {
                        assert_ne!(target, dispatched);
                        let seen = seen.lock().expect("the scores");
                        assert!(seen.iter().any(|&(id, sent)| {
                            id == dispatched
                                && (sent.requests, sent.tokens, sent.unsized_requests)
                                    == (1, u64::from(TOKENS), 0)
                        }));
                        assert!(seen.contains(&(target, CreditedDispatches::default())));
                    } else {
                        first = Some(target);
                        let lease_id = answer.leases[0].id;
                        frames
                            .send(GatewayFrame {
                                seq: 1,
                                frame: Some(gateway_frame::Frame::Dispatched(Dispatched {
                                    lease_id,
                                    side: Side::Single.into(),
                                    since_decision_micros: 83,
                                    ..Dispatched::default()
                                })),
                                ..GatewayFrame::default()
                            })
                            .await
                            .expect("a dispatch");
                        let gateway = GatewayId(fingerprint(GATEWAY.as_bytes()));
                        while !LeasesStore::get(&*store, gateway, LeaseId(lease_id)).is_some_and(
                            |record| matches!(record.state, LeaseState::Dispatched { .. }),
                        ) {
                            tokio::time::sleep(Duration::from_millis(3)).await;
                        }
                    }
                }
            })
            .await
            .expect("the decisions and dispatch");
            stop.send(()).expect("the serve is listening for the stop");
        }
    );
    served.expect("a clean stop");
}
