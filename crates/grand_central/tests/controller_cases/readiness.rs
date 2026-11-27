//! Readiness through a drain and a shutdown, and a controller that cannot
//! read its lease.

use std::time::{Duration, Instant};

use smg_grand_central::service::{
    controller::{Controller, ElectionConfig, LeasePlace, Scope, State},
    elector::{FileLease, LeaseBackend},
    proto::{
        controller_frame, decide_answer, gateway_frame, grand_central_client::GrandCentralClient,
        DecideRequest, GatewayFrame, Hello, LeaderRequest, RequestFacts,
    },
};
use tokio::{
    net::TcpStream,
    sync::{mpsc, oneshot},
    time::timeout,
};
use tonic::{codegen::tokio_stream::wrappers::ReceiverStream, Code};
use tonic_health::pb::health_check_response::ServingStatus;

use crate::{config, health, net::http_get, stating};

/// A drain under a gateway's open decide streams, one that asked and one
/// that never did. Each ends `UNAVAILABLE` when the listeners stop
/// accepting, and the serve returns then: a gateway that closes none of its
/// sides does not hold the drain to the end of its grace period.
#[tokio::test]
async fn a_drain_ends_the_open_decide_streams_and_returns_inside_its_grace_period() {
    const GATEWAY: &str = "gateway-a";
    const SETTLE: Duration = Duration::from_millis(500);
    let mut config = config();
    config.shutdown_settle = SETTLE;
    config.shutdown_grace = Duration::from_secs(60);
    let mut controller = Controller::bind(config).await.expect("bind");
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
        let mut client = GrandCentralClient::connect(format!("http://{addr}"))
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
        let welcome = timeout(Duration::from_secs(20), session.message())
            .await
            .expect("a Welcome within the case's patience")
            .expect("a frame, not a status")
            .expect("a frame, not the stream's end");
        assert!(
            matches!(welcome.frame, Some(controller_frame::Frame::Welcome(_))),
            "a Welcome answers a Hello: {welcome:?}"
        );

        // The stream that asked: a request with no budget, answered here.
        let (requests, sent) = mpsc::channel(1);
        let mut asked = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a present gateway's stream opens")
            .into_inner();
        requests
            .send(DecideRequest {
                request_id: vec![1],
                facts: Some(RequestFacts {
                    model: "model-a".to_owned(),
                    ..RequestFacts::default()
                }),
                constraints: None,
                again: false,
                ..Default::default()
            })
            .await
            .expect("the stream takes the request");
        let answer = timeout(Duration::from_secs(20), asked.message())
            .await
            .expect("an answer within the case's patience")
            .expect("an answer, not a status")
            .expect("an answer, not the stream's end");
        assert!(
            matches!(answer.answer, Some(decide_answer::Answer::Busy(_))),
            "a Busy: {answer:?}"
        );
        let (silent, sent) = mpsc::channel::<DecideRequest>(1);
        let mut idle = client
            .decide(stating(Some(GATEWAY), ReceiverStream::new(sent)))
            .await
            .expect("a present gateway's stream opens")
            .into_inner();

        let signalled = Instant::now();
        stop.send(()).expect("the serve is listening for the stop");
        for stream in [&mut asked, &mut idle] {
            let ended = timeout(Duration::from_secs(20), stream.message())
                .await
                .expect("the stream's end within the case's patience")
                .expect_err("a status ends the stream");
            assert_eq!(
                (ended.code(), ended.message()),
                (
                    Code::Unavailable,
                    "this controller closes its listeners: open the decide stream on the leader (ask Leader)"
                )
            );
        }
        // What the gateway holds stays open past the serve's end.
        let held = (client, frames, session, requests, asked, silent, idle);
        (signalled, held)
    };
    let (served, (signalled, _held)) = tokio::join!(serve, probe);
    served.expect("a clean stop");
    let took = signalled.elapsed();
    assert!(took >= SETTLE, "the settle window held: {took:?}");
    assert!(
        took < Duration::from_secs(20),
        "the drain did not wait for the grace: {took:?}"
    );
}

#[tokio::test]
async fn readiness_fails_at_the_signal_and_the_listeners_settle_before_they_close() {
    let mut config = config();
    config.shutdown_settle = Duration::from_secs(1);
    config.shutdown_grace = Duration::from_secs(10);
    let controller = Controller::bind(config).await.expect("bind");
    let addresses = controller.addresses();
    let mut readiness = controller.readiness();
    let gateway = addresses.of(Scope::Gateway).expect("the gateway listener");

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        let (status, _) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(status, 200);

        let signalled = Instant::now();
        stop.send(()).expect("the serve is listening for the stop");
        assert_eq!(
            readiness.wait_for(State::Draining).await,
            Some(State::Draining)
        );

        // Inside the settle window: not ready, still accepting, still alive.
        let (status, body) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(
            (status, body.as_str()),
            (503, r#"{"status":"not ready","reason":"draining"}"#)
        );
        let (status, body) = http_get(addresses.admin, "/liveness").await.expect("GET");
        assert_eq!((status, body.as_str()), (200, "OK"));
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::NotServing));
        let (status, body) = http_get(addresses.admin, "/metrics").await.expect("GET");
        assert_eq!(status, 200);
        assert!(
            body.contains("smg_grand_central_state{state=\"draining\"} 1"),
            "{body}"
        );
        signalled
    };
    let (served, signalled) = tokio::join!(serve, probe);
    served.expect("a clean stop");
    let took = signalled.elapsed();
    assert!(
        took >= Duration::from_secs(1),
        "the settle window held: {took:?}"
    );
    assert!(
        took < Duration::from_secs(10),
        "the drain did not wait for the grace: {took:?}"
    );

    assert!(
        TcpStream::connect(gateway).await.is_err(),
        "the gateway listener closed"
    );
    assert!(
        TcpStream::connect(addresses.admin).await.is_err(),
        "the admin listener closed"
    );
}

/// A controller that cannot read its lease is not ready: its probes answer
/// starting and `Leader` names nobody, however long it runs. Once the lease
/// reads, the next step takes it and the probes turn ready. A rollout stops
/// at a replica that cannot reach its lease.
#[tokio::test]
async fn a_controller_that_cannot_read_its_lease_is_not_ready_until_it_can() {
    let dir = tempfile::tempdir().expect("a temp dir");
    // A directory where the lease file belongs: every look fails.
    let lease = dir.path().join("lease");
    std::fs::create_dir(&lease).expect("a directory in the lease's place");
    let mut config = config();
    // No settle window: the stop's release is all the serve waits for.
    config.shutdown_settle = Duration::ZERO;
    config.election = ElectionConfig::default();
    config.election.lease = LeasePlace::File(lease.clone());
    config.election.settings.duration = Duration::from_millis(600);
    config.election.settings.renew_period = Duration::from_millis(50);
    config.election.settings.renew_deadline = Duration::from_millis(400);
    config.election.advertise = None;
    let controller = Controller::bind(config).await.expect("bind");
    let addresses = controller.addresses();
    let gateway = addresses.of(Scope::Gateway).expect("the gateway listener");
    let fence = controller.fence();
    let mut readiness = controller.readiness();

    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let probe = async {
        // Ten renew periods of failed looks: the listeners answer, and the
        // controller is not ready.
        tokio::time::sleep(Duration::from_millis(500)).await;
        assert_eq!(readiness.state(), State::Starting);
        let (status, body) = http_get(addresses.admin, "/readiness").await.expect("GET");
        assert_eq!(
            (status, body.as_str()),
            (503, r#"{"status":"not ready","reason":"starting"}"#)
        );
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::NotServing));
        assert_eq!(fence.leader_term(Instant::now()), None);
        let mut client = GrandCentralClient::connect(format!("http://{gateway}"))
            .await
            .expect("the gateway listener accepts");
        let nobody = client
            .leader(LeaderRequest::default())
            .await
            .expect("Leader is served while starting")
            .into_inner();
        assert!(!nobody.is_leader);
        assert_eq!(
            (nobody.controller_id.as_str(), nobody.address.as_str()),
            ("", "")
        );

        // The lease reads again: the next step takes it.
        std::fs::remove_dir(&lease).expect("the directory goes");
        assert_eq!(readiness.wait_for(State::Leader).await, Some(State::Leader));
        assert_eq!(fence.leader_term(Instant::now()), Some(1));
        assert_eq!(health(gateway, "").await, Ok(ServingStatus::Serving));
        let leader = client
            .leader(LeaderRequest::default())
            .await
            .expect("Leader is served")
            .into_inner();
        assert!(leader.is_leader);
        assert_eq!(leader.controller_id, "test-controller");
        assert_eq!(leader.address, gateway.to_string());
        assert_eq!(leader.term, 1);

        stop.send(()).expect("the serve is listening for the stop");
    };
    let (served, ()) = tokio::join!(serve, probe);
    served.expect("a clean stop");

    // The stop released the lease: the file names nobody, under the term.
    let record = FileLease::new(&lease)
        .read()
        .expect("the lease file reads")
        .expect("the lease file holds a record");
    assert!(record.is_released(), "{record:?}");
    assert_eq!(record.term, 1);
}
