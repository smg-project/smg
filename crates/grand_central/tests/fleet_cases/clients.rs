//! The clients of a started controller: the blocking client, a gateway's
//! link and its leases, and a client at a replica that does not lead.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use smg_grand_central::service::{
    client::{self, blocking, Answer, Client, DecisionUnavailable, Link},
    controller::State,
    proto::{
        completed::Class, decision::Outcome, gateway_frame, released, worker_event, Decision,
        Dispatched, Hello, Lease, Released, ShedReason, WatchWorkersRequest,
    },
};
use tokio::{
    sync::{oneshot, Barrier},
    task::JoinSet,
};
use tonic::{transport::Endpoint, Code};

use crate::{
    ask,
    fleet::{within, Harness, Replica, ROOM},
    keyed, linked_fleet, linked_pick, nth, session_of, ASKS, CASE, MODEL,
};

/// The blocking form of the reference client asks the same started
/// controller from a thread that runs no runtime, and holds the same floor.
/// Its watch tells the controller's refusal of a request and, of one the
/// controller takes, the snapshot first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_blocking_client_asks_a_started_controller_from_a_thread_of_its_own() {
    let case = async {
        let harness = Harness::start().await.expect("the harness starts");
        let address = format!("http://{}", harness.gateway());
        let (answered, answer) = oneshot::channel();
        std::thread::spawn(move || {
            let asked =
                blocking::Client::connect(&address, "blocking-gateway").and_then(|mut client| {
                    let leader = client.leader()?;
                    let subject = client
                        .welcome()
                        .principal
                        .as_ref()
                        .map(|told| told.subject.clone());
                    let decided = client.decide(ask(b"blocking"))?;
                    // A resume under a term with no place is refused by its field.
                    let unplaced = WatchWorkersRequest {
                        resume_term: 1,
                        ..WatchWorkersRequest::default()
                    };
                    let refused = match client.watch_workers(unplaced) {
                        Err(client::Error::Status(status)) => {
                            Some((status.code(), status.message().to_owned()))
                        }
                        _ => None,
                    };
                    let mut watch = client.watch_workers(WatchWorkersRequest::default())?;
                    let told = client.next_of(&mut watch)?;
                    Ok((leader, client.term_floor(), subject, decided, refused, told))
                });
            let _ = answered.send(asked.map_err(|error| error.to_string()));
        });
        let (leader, floor, subject, decided, refused, told) = tokio::time::timeout(ROOM, answer)
            .await
            .expect("the blocking client answers inside its bound")
            .expect("the client's thread sends its answer")
            .expect("the blocking client's calls");
        assert!(leader.is_leader, "a lone controller leads: {leader:?}");
        assert_eq!(leader.controller_id, "fleet-controller");
        assert_eq!(floor, leader.term, "the floor is the leader's term");
        assert_eq!(
            subject.as_deref(),
            Some("blocking-gateway"),
            "the welcome names the gateway the client states"
        );
        // No worker serves the model: the controller decides over no list,
        // and the answer is a shed for want of endpoints under the stage
        // `none`, whatever document is applied.
        assert!(
            matches!(&decided, Answer::Decision(Decision { outcome: Some(Outcome::Shed(shed)), .. }, _)
                if shed.reason() == ShedReason::NoEndpoints && shed.emptied_by == "none"),
            "{decided:?}"
        );
        // The watch: an open the controller refuses tells its status, and
        // the first event of one it takes is the snapshot of no worker.
        assert!(
            matches!(&refused, Some((Code::InvalidArgument, said)) if said.contains("resume_term")),
            "{refused:?}"
        );
        let told = told.expect("the watch tells a snapshot first");
        assert!(
            matches!(&told.event, Some(worker_event::Event::Snapshot(page))
                if page.last && page.workers.is_empty()),
            "{told:?}"
        );
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

async fn leases_before_expiry(
    harness: &Harness,
    link: &Link,
    gateway: &str,
    left: usize,
    before: Instant,
) -> Result<(), String> {
    let span = Duration::from_millis(u64::from(link.welcome().dispatch_ttl_ms));
    within(
        "the store reaches the reported lease count before expiry",
        span.saturating_sub(before.elapsed()),
        || before.elapsed() < span && harness.leases(gateway) == left,
    )
    .await
    .map(|_| ())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_answers_sixty_four_asks_and_reports_the_end_of_each_lease() {
    const GATEWAY: &str = "fleet-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let mut asks = JoinSet::new();
        let count = usize::try_from(ASKS).expect("the number of asks fits");
        let ready = Arc::new(Barrier::new(count + 1));
        let before = Instant::now();
        for n in 0..ASKS {
            let link = link.clone();
            let ready = Arc::clone(&ready);
            asks.spawn(async move {
                let session = session_of(&n.to_string(), MODEL);
                ready.wait().await;
                let lease = linked_pick(&link, keyed(nth("link", n), session))
                    .await
                    .expect("the controller answers with a pick");
                assert_eq!(lease.session_key, session, "each ask gets its own decision");
                (lease.id, lease.side)
            });
        }
        ready.wait().await;
        let mut leases = HashMap::new();
        while let Some(answer) = asks.join_next().await {
            let (id, side) = answer.expect("the caller gets its answer");
            assert!(
                leases.insert(id, side).is_none(),
                "a distinct lease for every ask"
            );
        }
        assert_eq!(harness.leases(GATEWAY), count);
        for (lease_id, side) in leases {
            link.queue_frame(gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease_id)),
                reason: released::Reason::ClientGone.into(),
                side,
            }));
        }
        leases_before_expiry(&harness, &link, GATEWAY, 0, before)
            .await
            .expect("the reports end the leases");
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// A request can outlast the session its decision came over. The gateway
/// dials its link again, and what the request reports then reaches the
/// controller on the link's next session: a dispatched lease, which no
/// deadline ends, ends at that report.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dispatched_lease_is_ended_by_a_report_on_the_links_next_session() {
    const GATEWAY: &str = "fleet-redial";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let endpoint =
            Endpoint::from_shared(format!("http://{}", harness.gateway())).expect("a URI");
        // A request in flight holds a clone of the link.
        let held = link.clone();
        let sent = |lease: Lease| {
            gateway_frame::Frame::Dispatched(Dispatched {
                lease_id: lease.id,
                target_index: 0,
                side: lease.side,
                since_decision_micros: 0,
            })
        };
        let ended = |lease: Lease| {
            gateway_frame::Frame::Released(Released {
                what: Some(released::What::LeaseId(lease.id)),
                reason: released::Reason::ClientGone.into(),
                side: lease.side,
            })
        };

        // The controller ends the session under a dispatched request: a
        // second hello on one stream is refused.
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        held.queue_frame(sent(first));
        within("the store reads the first lease dispatched", ROOM, || {
            harness.dispatched(GATEWAY, first.id)
        })
        .await
        .expect("the session said the dispatch");
        link.queue_frame(gateway_frame::Frame::Hello(Hello::default()));
        tokio::time::timeout(ROOM, link.closed())
            .await
            .expect("the session ends");
        assert!(!held.stands());

        // The request ends with no session standing. Its report waits, and
        // the link's next session says it.
        held.queue_frame(ended(first));
        assert_eq!(harness.leases(GATEWAY), 1, "no session said the end");
        Box::pin(link.redial(endpoint.clone(), GATEWAY))
            .await
            .expect("the link's next session");
        within("the report that waited ends the lease", ROOM, || {
            harness.leases(GATEWAY) == 0
        })
        .await
        .expect("the next session said the report");

        // A redial over a session that stands: the next session takes its
        // place, and the gateway's lease with it.
        let second = linked_pick(&held, ask(b"second"))
            .await
            .expect("a pick on the next session");
        held.queue_frame(sent(second));
        within("the store reads the second lease dispatched", ROOM, || {
            harness.dispatched(GATEWAY, second.id)
        })
        .await
        .expect("the session said the dispatch");
        Box::pin(link.redial(endpoint, GATEWAY))
            .await
            .expect("the link's third session");
        assert_eq!(
            harness.leases(GATEWAY),
            1,
            "the lease stands through the redial"
        );
        held.queue_frame(ended(second));
        within(
            "the report on the third session ends the lease",
            ROOM,
            || harness.leases(GATEWAY) == 0,
        )
        .await
        .expect("the third session said the report");
        assert!(held.stands());
        assert_eq!(held.frames_dropped(), 0);
        drop((link, held));
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_releases_the_lease_of_a_zero_budget_ask() {
    const GATEWAY: &str = "fleet-late-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let before = Instant::now();
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        assert!(matches!(
            link.ask(ask(b"late"), Duration::ZERO).await,
            Err(DecisionUnavailable::Late)
        ));
        let last = linked_pick(&link, ask(b"last")).await.expect("a last pick");
        // The three asks run on the same decider. The gap proves that the
        // middle ask booked a lease even if its release has arrived already.
        assert_eq!(last.id, first.id + 2);
        leases_before_expiry(&harness, &link, GATEWAY, 2, before)
            .await
            .expect("only the first and last leases remain");
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// A frame the call refuses ends the stream it is sent on, and a link's
/// stream is every caller's. The link refuses such a frame at its own door:
/// nothing is sent, the link stands, and the next ask is answered on the
/// stream that stood.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_link_refuses_a_frame_out_of_the_calls_ranges_and_stands() {
    const GATEWAY: &str = "fleet-ranged-link";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 1))
            .await
            .expect("a ready fleet");
        let first = linked_pick(&link, ask(b"first"))
            .await
            .expect("a first pick");
        let mut wrong = ask(b"wrong");
        wrong.facts.as_mut().expect("the ask states facts").phase = 99;
        let refused = link.ask(wrong, ROOM).await;
        assert!(
            matches!(&refused, Err(DecisionUnavailable::Unsent(error))
                if error.to_string().contains("facts.phase")),
            "{refused:?}"
        );
        assert!(link.stands(), "the link stands");
        let last = linked_pick(&link, ask(b"last"))
            .await
            .expect("a pick after the refusal");
        // Both picks ran on the same decider: no lease was booked between
        // them, so the refused frame reached no decider.
        assert_eq!(last.id, first.id + 1);
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handle_over_a_link_ends_a_lease_its_alternate_served() {
    use smg_grand_central::{
        engine::{InProcess, Request, RequestFacts},
        tracking::{DecisionGuard, Outcome, Side, TransportClass},
    };

    const GATEWAY: &str = "fleet-handle";
    let case = async {
        let (harness, link) = Box::pin(linked_fleet(GATEWAY, 2))
            .await
            .expect("a ready fleet");
        for id in ["held-0", "held-1", "held-2"] {
            let asked = Request::of(id.as_bytes(), {
                let mut facts = RequestFacts::default();
                facts.model = MODEL;
                facts.budget_micros = 5_000_000;
                facts
            });
            let started = Instant::now();
            let held = Box::pin(DecisionGuard::<&InProcess>::asked(
                &link, &asked, started, ROOM,
            ))
            .await;
            let mut handle = held.expect("the controller's decision");
            let taken = handle.take(0).expect("a picked worker");
            let alternate = handle.decision().pick().expect("a pick").targets.as_slice()[1];
            let (picked, _) = harness
                .named(taken.id.0)
                .expect("the picked worker is in the fleet");
            let (served, _) = harness
                .named(alternate.id.0)
                .expect("the alternate is in the fleet");
            assert_ne!(served, picked);
            assert_eq!(harness.leases(GATEWAY), 1, "the controller filed the lease");
            let sent = Instant::now();
            handle.dispatched(sent);
            let (class, frames) = harness
                .serve_request(served, id.to_owned())
                .await
                .expect("the worker answers");
            assert_eq!(class, Class::Ok);
            assert!(frames > 0);
            let ended = Instant::now();
            let mut outcome =
                Outcome::reported(0, TransportClass::Ok).with_served(Side::Single, alternate.id);
            outcome.tokens_in = 8;
            outcome.tokens_out = frames;
            outcome.duration_micros =
                u32::try_from(ended.duration_since(sent).as_micros()).unwrap_or(u32::MAX);
            handle.completed(&outcome, ended);
            leases_before_expiry(&harness, &link, GATEWAY, 0, started)
                .await
                .expect("the completion ends the lease");
        }
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A request that names its worker is answered that worker over a link, by
/// the number its host states at the door's ask. Under the stock hash ring
/// document, whose binding names a target source, each number of a fleet of
/// three is answered one worker and no alternate, the same at each ask
/// whatever the request's id, and the three numbers are answered the three
/// workers in the order they registered. A number past the fleet and a text
/// that is no number are answered a shed with no endpoint. A request that
/// names no worker is answered a pick with its alternates, as it was.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handle_over_a_link_is_answered_the_worker_its_request_names() {
    use smg_grand_central::{
        engine::{Decision as Decided, InProcess, Request, RequestFacts, ShedReason as Why},
        tracking::DecisionGuard,
    };

    const GATEWAY: &str = "fleet-named";
    let case = async {
        let (mut harness, link) = Box::pin(linked_fleet(GATEWAY, 3))
            .await
            .expect("a ready fleet");
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let asks = [
            ("named-0a", Some(Some(0))),
            ("named-1a", Some(Some(1))),
            ("named-2a", Some(Some(2))),
            ("named-0b", Some(Some(0))),
            ("named-1b", Some(Some(1))),
            ("named-2b", Some(Some(2))),
            ("named-3", Some(Some(3))),
            ("named-text", Some(None)),
            ("named-none", None),
        ];
        // The engine an ask is answered and how many targets its pick
        // names, or the reason of its shed.
        let mut answers = Vec::with_capacity(asks.len());
        for (id, named) in asks {
            let asked = Request::of(id.as_bytes(), {
                let mut facts = RequestFacts::default();
                facts.model = MODEL;
                facts.budget_micros = 5_000_000;
                facts
            });
            let started = Instant::now();
            let held = Box::pin(DecisionGuard::<&InProcess>::asked_naming(
                &link, &asked, named, started, ROOM,
            ))
            .await;
            let handle = held.expect("the controller's decision");
            let answer = match handle.decision() {
                Decided::Pick(pick) => {
                    let first = pick.targets.as_slice()[0];
                    let (engine, _) = harness
                        .named(first.id.0)
                        .expect("the picked worker is in the fleet");
                    Ok((engine, pick.targets.len()))
                }
                Decided::Shed(shed) => Err(shed.reason),
                Decided::Queue(_) => panic!("{id} is queued"),
            };
            answers.push(answer);
        }
        let alone = |engine: usize| -> Result<(usize, usize), Why> { Ok((engine, 1)) };
        let twice = [alone(0), alone(1), alone(2), alone(0), alone(1), alone(2)];
        assert_eq!(answers[..6], twice, "its worker alone, at each ask");
        let shed = [Err(Why::NoEndpoints), Err(Why::NoEndpoints)];
        assert_eq!(answers[6..8], shed, "a number that reaches no worker");
        let unnamed = answers[8];
        assert!(
            matches!(unnamed, Ok((_, targets)) if targets > 1),
            "{unnamed:?}"
        );
        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A single request's hard demands cross a link, and the controller holds
/// the request to the workers whose registrations meet them.
///
/// Three engines register over the control listener. One states a runtime
/// and a window, and processes media by reference by its registering
/// party's pipeline. One states another runtime and a pipeline of its own.
/// One states none of those, and that it takes images.
///
/// The link's session is welcomed on protocol 2, so the handle states each
/// demand on its frame. Every ask of a demand a worker meets is a pick, and
/// the picks name the workers that meet it and no other. A runtime no
/// worker states is shed at every ask, and so is an input past the window
/// of its runtime's one worker: each with no endpoint, under its demand's
/// stage as the handle reads it. A demand for vision crosses too: every
/// pick of it names the engine whose registration says that it takes
/// images. Beside a runtime that engine does not state, it is shed too.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_handle_over_a_link_is_held_to_the_workers_that_meet_its_requests_demands() {
    use std::collections::BTreeSet;

    use smg_grand_central::{
        engine::{
            Constraints, Decision as Decided, Demands, InProcess, References, Request,
            RequestFacts, Role, ShedReason as Why,
        },
        service::{controller::PROTOCOL_VERSION, proto},
        tracking::DecisionGuard,
    };

    use crate::FLEET;

    const GATEWAY: &str = "fleet-demands";
    /// The asks a demand is given.
    const ASKS: usize = 12;
    let case = async {
        let mut harness = Harness::start().await.expect("a started controller");
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        let stated = [
            Some(proto::DemandFacts {
                runtime: 7,
                references: true,
                host_pipeline: true,
                window: 4_096,
                decoding: 0,
            }),
            Some(proto::DemandFacts {
                runtime: 8,
                references: true,
                host_pipeline: false,
                window: 0,
                decoding: 0,
            }),
            None,
        ];
        for (engine, block) in stated.into_iter().enumerate() {
            harness.states(block);
            // The last engine's registration says that it takes images.
            harness.sees((engine == 2).then_some(true));
            harness
                .register(&FLEET[engine..=engine])
                .await
                .expect("the engine registers");
        }
        harness.feed();
        within("the workers are ready", ROOM, || {
            harness
                .pool(MODEL, Role::Single)
                .is_ok_and(|pool| pool.len() == 3 && pool.iter().all(|row| row.healthy))
        })
        .await
        .expect("three ready workers");
        let endpoint = Endpoint::from_shared(format!("http://{}", harness.gateway()))
            .expect("the gateway listener's address");
        let link = Box::pin(Link::connect(endpoint, GATEWAY))
            .await
            .expect("the link's session opens");
        assert_eq!(link.welcome().protocol, PROTOCOL_VERSION);

        let none = Demands::default();
        let mut a_runtime = none;
        a_runtime.runtime = 7;
        let mut at_worker = none;
        at_worker.references = References::Worker;
        let mut past = none;
        past.input_window = 4_097;
        let mut unstated = none;
        unstated.runtime = 9;
        let mut past_its_one = none;
        past_its_one.runtime = 7;
        past_its_one.input_window = 4_097;
        let mut vision = none;
        vision.vision = true;
        let mut vision_at_a_runtime = vision;
        vision_at_a_runtime.runtime = 7;
        // What a demand's asks are answered: the engines their picks name, or
        // the reason and the stage of a shed.
        let demands = [
            ("no demand", none, Ok(BTreeSet::from([0, 1, 2]))),
            ("a runtime", a_runtime, Ok(BTreeSet::from([0]))),
            ("a pipeline of its own", at_worker, Ok(BTreeSet::from([1]))),
            ("an input past a window", past, Ok(BTreeSet::from([1, 2]))),
            (
                "a runtime none states",
                unstated,
                Err((Why::NoEndpoints, "wires")),
            ),
            (
                "an input past the window of its runtime's one worker",
                past_its_one,
                Err((Why::NoEndpoints, "context-window")),
            ),
            ("vision", vision, Ok(BTreeSet::from([2]))),
            (
                "vision and a runtime its engine does not state",
                vision_at_a_runtime,
                Err((Why::NoEndpoints, "vision")),
            ),
        ];
        let asked = |id: &'static [u8], seed: u64, demands: Demands| {
            let mut request = Request::of(id, {
                let mut facts = RequestFacts::default();
                facts.model = MODEL;
                facts.seed = seed;
                facts.budget_micros = 5_000_000;
                facts
            });
            request.constraints = {
                let mut constraints = Constraints::default();
                constraints.demands = demands;
                constraints
            };
            request
        };
        let mut seed = 0;
        for (what, demanded, wanted) in demands {
            // Every ask counts: the engines the picks name, and each shed.
            let mut named = BTreeSet::new();
            let mut shed = Vec::new();
            for _ in 0..ASKS {
                seed += 1;
                let request = asked(b"demanded", seed, demanded);
                let held = Box::pin(DecisionGuard::<&InProcess>::asked(
                    &link,
                    &request,
                    Instant::now(),
                    ROOM,
                ))
                .await;
                let handle = held.unwrap_or_else(|refused| panic!("{what}: {refused:?}"));
                match handle.decision() {
                    Decided::Pick(pick) => {
                        for target in pick.targets.as_slice() {
                            let (engine, _) = harness
                                .named(target.id.0)
                                .expect("the picked worker is in the fleet");
                            named.insert(engine);
                        }
                    }
                    Decided::Shed(told) => shed.push((told.reason, told.emptied_by)),
                    Decided::Queue(_) => panic!("{what} is queued"),
                }
            }
            let answered = match wanted {
                Ok(engines) => shed.is_empty() && named == engines,
                Err(why) => named.is_empty() && shed == [why; ASKS],
            };
            assert!(answered, "{what}: picks {named:?}, sheds {shed:?}");
        }

        assert!(link.stands());
        drop(link);
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A replica that does not lead refuses the client's session and names the
/// leader: two controllers elect over one lease file, and the client dials
/// each.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_client_at_a_replica_that_does_not_lead_is_told_the_leader() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let mut leader = Replica::start("replica-a", Some(&lease))
            .await
            .expect("the first replica starts");
        leader
            .reaches(State::Leader)
            .await
            .expect("the first replica leads");
        let mut follower = Replica::start("replica-b", Some(&lease))
            .await
            .expect("the second replica starts");
        follower
            .reaches(State::Follower)
            .await
            .expect("the second replica follows");

        let at_follower = format!("http://{}", follower.gateway());
        let refused = tokio::time::timeout(ROOM, Client::connect(&at_follower, "pair-gateway"))
            .await
            .expect("the follower answers inside its bound");
        let address = match refused {
            Ok(welcomed) => Err(format!("it welcomed the gateway: {welcomed:?}")),
            Err(client::Error::NotLeader(address)) => Ok(address),
            Err(other) => Err(other.to_string()),
        }
        .expect("a replica that does not lead names the leader");
        assert_eq!(
            address,
            leader.gateway().to_string(),
            "the address is the leader's gateway listener"
        );

        let welcomed = tokio::time::timeout(
            ROOM,
            Client::connect(&format!("http://{address}"), "pair-gateway"),
        )
        .await
        .expect("the leader answers inside its bound")
        .expect("the leader welcomes the gateway its follower sent on");
        assert_eq!(welcomed.welcome().controller_id, "replica-a");
        drop(welcomed);

        follower.stop().await.expect("the follower stops clean");
        leader.stop().await.expect("the leader stops clean");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}
