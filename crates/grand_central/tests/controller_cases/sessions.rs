//! A gateway's session through the bound controller: the grace period behind
//! its stream's end, a term floor above the term, and a serve dropped under
//! an open session.

use std::time::{Duration, Instant};

use smg_grand_central::{
    engine::{fingerprint, WorkerId},
    service::{
        controller::{Controller, Scope, State},
        proto::{
            controller_frame, gateway_frame, grand_central_client::GrandCentralClient,
            GatewayFrame, Hello,
        },
    },
    store::{GatewayId, LeaseRecord, LeaseState, LeasesStore, PickRecord},
    tracking::{Lease, LeaseId},
};
use tokio::{
    sync::{mpsc, oneshot},
    time::timeout,
};
use tonic::{codegen::tokio_stream::wrappers::ReceiverStream, Code};

use crate::{config, stating};

/// A gateway's session through the bound controller. It is welcomed under
/// the leader's term, and the run loop's sweep lets it go a grace period
/// after its stream's end: the lease it held is released then, not before.
#[tokio::test]
async fn a_gateway_gone_past_its_grace_period_is_let_go_by_the_run_loop() {
    const GATEWAY: &str = "gateway-a";
    let mut controller = Controller::bind(config()).await.expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let store = controller.store();
    // The store keys a gateway by its subject's fingerprint.
    let gateway = GatewayId(fingerprint(GATEWAY.as_bytes()));

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
        let (frames, sent) = mpsc::channel(4);
        let mut answers = client
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
        let answer = timeout(Duration::from_secs(20), answers.message())
            .await
            .expect("a Welcome within the case's patience")
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        assert_eq!(answer.term, 1, "the first take's term: {answer:?}");
        let grace = match answer.frame {
            Some(controller_frame::Frame::Welcome(welcome)) => {
                Duration::from_millis(u64::from(welcome.grace_period_ms))
            }
            other => panic!("a Welcome answers a Hello: {other:?}"),
        };

        // A lease the gateway holds, as a decision records it.
        let now = Instant::now();
        store
            .issue(LeaseRecord {
                lease: {
                    let mut lease = Lease::default();
                    lease.id = LeaseId(1);
                    lease
                },
                gateway,
                request: 1,
                target: WorkerId(7),
                picked: PickRecord::default(),
                decode: None,
                untold: false,
                term: 1,
                issued_at: now,
                dispatch_deadline: now + Duration::from_secs(600),
                state: LeaseState::ISSUED,
            })
            .expect("a lease under the leader's term");
        assert_eq!(store.live(gateway), 1);

        // The stream ends. The gateway stays present for the grace period,
        // then the sweep releases what it held.
        drop(frames);
        drop(answers);
        let ended = Instant::now();
        while store.live(gateway) > 0 {
            assert!(
                ended.elapsed() < grace + Duration::from_secs(20),
                "the lease stands {:?} after the stream's end",
                ended.elapsed()
            );
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        assert!(
            ended.elapsed() >= grace,
            "let go {:?} after the stream's end, inside the grace period of {grace:?}",
            ended.elapsed()
        );

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// A gateway's term floor above the leader's term, through the bound
/// controller with the election's default: one replica alone over its own
/// lease. The `Hello` is refused as before, and the floor reaches the
/// election: the replica gives the lease up and, alone, takes it again at
/// its next step above the floor. The gateway's next `Hello` under the same
/// floor is welcomed under that term.
#[tokio::test]
async fn a_gateways_term_floor_above_the_term_moves_the_lone_replica_above_it() {
    const GATEWAY: &str = "gateway-a";
    const FLOOR: u64 = 7;
    let mut controller = Controller::bind(config()).await.expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let addr = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let client = GrandCentralClient::connect(format!("http://{addr}"))
            .await
            .expect("the listener accepts");
        // One `Hello` under the floor on a stream of its own: the term of
        // the Welcome that answers it, `None` for another frame or the
        // stream's end, or the status that ended the stream.
        let hello = || {
            let mut client = client.clone();
            async move {
                let (frames, sent) = mpsc::channel(4);
                let mut answers = client
                    .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
                    .await?
                    .into_inner();
                frames
                    .send(GatewayFrame {
                        seq: 0,
                        term_floor: FLOOR,
                        frame: Some(gateway_frame::Frame::Hello(Hello {
                            gateway_id: GATEWAY.to_owned(),
                            protocol_min: 1,
                            protocol_max: 1,
                            ..Hello::default()
                        })),
                    })
                    .await
                    .expect("the stream takes the Hello");
                let answer = answers.message().await?;
                Ok::<_, tonic::Status>(answer.and_then(|answer| match answer.frame {
                    Some(controller_frame::Frame::Welcome(_)) => Some(answer.term),
                    _ => None,
                }))
            }
        };

        let refused = hello()
            .await
            .expect_err("a floor above the first take's term");
        assert_eq!(refused.code(), Code::FailedPrecondition, "{refused:?}");
        assert_eq!(
            refused.message(),
            "the gateway's term floor 7 is above this controller's term 1"
        );

        // Until the retake a Hello is refused again or told NotLeader.
        let said = Instant::now();
        let term = loop {
            if let Ok(Some(term)) = hello().await {
                break term;
            }
            assert!(
                said.elapsed() < Duration::from_secs(30),
                "no Welcome under the floor {:?} after it was said",
                said.elapsed()
            );
            tokio::time::sleep(Duration::from_millis(20)).await;
        };
        assert_eq!(term, FLOOR + 1, "the take after the floor's word");

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");
}

/// A host cancels a served controller by dropping it. A gateway's session
/// that is still open keeps its task, and the task keeps the service's
/// shared state with the election's door. The door holds the election's
/// words weakly: the election hears its handle go and closes the fence at
/// the drop, not at the session's close.
#[tokio::test]
async fn a_serve_dropped_under_an_open_session_closes_the_fence_at_the_drop() {
    const GATEWAY: &str = "gateway-a";
    let mut controller = Controller::bind(config()).await.expect("bind");
    controller
        .identity_as_told("gateway")
        .expect("a listener of the configuration");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let fence = controller.fence();

    let mut serve = Box::pin(controller.serve(std::future::pending::<()>()));
    let opened = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let addr = addresses
            .of(Scope::Gateway)
            .expect("every scope is mounted");
        let mut client = GrandCentralClient::connect(format!("http://{addr}"))
            .await
            .expect("the listener accepts");
        let (frames, sent) = mpsc::channel(4);
        let mut answers = client
            .session(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a stream")
            .into_inner();
        frames
            .send(GatewayFrame {
                seq: 0,
                term_floor: 0,
                frame: Some(gateway_frame::Frame::Hello(Hello {
                    gateway_id: GATEWAY.to_owned(),
                    protocol_min: 1,
                    protocol_max: 1,
                    ..Hello::default()
                })),
            })
            .await
            .expect("the stream takes the Hello");
        let welcome = answers
            .message()
            .await
            .expect("an answer")
            .expect("a frame");
        assert!(
            matches!(welcome.frame, Some(controller_frame::Frame::Welcome(_))),
            "{welcome:?}"
        );
        (client, frames, answers)
    };
    let held = tokio::select! {
        ended = &mut serve => Err(ended),
        held = opened => Ok(held),
    };
    // The session stays open to the end of the case.
    let _session = held.expect("the session opens while the controller serves");
    assert!(
        fence.leader_term(Instant::now()).is_some(),
        "the replica leads"
    );

    drop(serve);
    let dropped = Instant::now();
    while fence.leader_term(Instant::now()).is_some() {
        assert!(
            dropped.elapsed() < Duration::from_secs(5),
            "the fence is open {:?} after the drop, under the open session",
            dropped.elapsed()
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}
