//! The membership door end to end (design 5.1 and 5.2): the control scope's
//! three registration calls against this package's binary as a process, two
//! replicas over one lease file, and against the same serve in the case's
//! own process, where the store's handle is read back.
//!
//! No stub server: the client is the generated one and the server is the
//! binary. The one engine that answers is the mock worker's, in the case's
//! process: the leader is given its address through the door and nothing
//! else, and its own load monitor's counter says it polled it.
//!
//! The door opens to a caller with a principal: the control listener is one
//! its operator named, and every call states the caller's name. The
//! listener's word for the service's name says where the door is open.
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "this binary reads one admin page; the other helpers are the listeners' tests'"
)]
#[path = "common/net.rs"]
mod net;
#[expect(
    dead_code,
    reason = "the failover helpers are the replicas case's and the election cells'"
)]
#[path = "common/replicas.rs"]
mod replicas;

use std::{
    ffi::OsStr,
    io::Write,
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use net::{http_get, loopback};
use openai_protocol::worker::RuntimeType;
use replicas::{Lease, Record, Replica};
use smg_policy_controller::{
    service::{
        controller::{Config, Controller, ElectionConfig, Listener, Scope, State, SERVICE_NAME},
        elector::ElectionSettings,
        proto::{
            self, policy_controller_client::PolicyControllerClient, register_worker_request::Mode,
            worker_event::Event,
        },
    },
    store::{WorkerStatus, WorkersStore},
};
use tokio::{net::TcpListener, sync::oneshot, task::JoinSet};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Code, Request};
use tonic_health::pb::{
    health_check_response::ServingStatus, health_client::HealthClient, HealthCheckRequest,
};

/// Long against a loaded host: the leader of the case's start leads to its
/// end. The renew deadline stays well under the kit's room for an agreement,
/// so a leader whose lead opens one deadline after its take is seen in time.
fn settings() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_secs(60),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_secs(10),
    }
}

async fn client(addr: SocketAddr) -> PolicyControllerClient<Channel> {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    PolicyControllerClient::new(channel)
}

/// `message` as a call that states the operator's name.
fn stated<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.metadata_mut().insert(
        "x-smg-principal",
        AsciiMetadataValue::from_static("operator-a"),
    );
    request
}

/// The listener's word for the service's name.
async fn word(addr: SocketAddr) -> ServingStatus {
    let channel = Channel::from_shared(format!("http://{addr}"))
        .expect("a uri")
        .connect()
        .await
        .expect("the listener accepts");
    HealthClient::new(channel)
        .check(HealthCheckRequest {
            service: SERVICE_NAME.to_owned(),
        })
        .await
        .expect("the health service knows the name")
        .into_inner()
        .status()
}

/// The listener's word once it is `wanted`, or its last word when `bound`
/// ran out: the word is told after the election's steps.
async fn word_within(addr: SocketAddr, wanted: ServingStatus, bound: Duration) -> ServingStatus {
    let started = Instant::now();
    loop {
        let now = word(addr).await;
        if now == wanted || started.elapsed() > bound {
            return now;
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

/// A mock gRPC engine of one rank, served until the returned sender drops.
async fn mock_engine() -> (SocketAddr, oneshot::Sender<()>) {
    let listener = TcpListener::bind(loopback()).await.expect("a port");
    let addr = listener.local_addr().expect("its address");
    let cfg = Arc::new(mock_worker::config::Config {
        grpc_base_port: addr.port(),
        grpc_count: 1,
        grpc_dp_size: 1,
        grpc_wire: mock_worker::config::SchedulerWire::Vllm,
        loads_like: mock_worker::engine::LoadsLike::Vllm,
        realistic: true,
        ..mock_worker::config::Config::default()
    });
    let (stop, stopped) = oneshot::channel();
    let mut served = JoinSet::new();
    served.spawn(mock_worker::grpc::serve_with_listener_until(
        cfg,
        listener,
        async move {
            let _ = stopped.await;
        },
    ));
    served.detach_all();
    (addr, stop)
}

/// A checked worker nobody dials: it stays pending.
fn spec(url: &str) -> proto::WorkerSpec {
    proto::WorkerSpec {
        url: url.to_owned(),
        wire: proto::WireMode::Grpc as i32,
        role: proto::Role::Single as i32,
        runtime: RuntimeType::Vllm.as_str().to_owned(),
        models: vec!["model-a".to_owned()],
        ..Default::default()
    }
}

/// The same with its checks off: ready from its registration.
fn unchecked(url: &str) -> proto::WorkerSpec {
    proto::WorkerSpec {
        health_policy: Some(proto::HealthPolicy {
            disabled: true,
            ..Default::default()
        }),
        ..spec(url)
    }
}

fn ask(spec: proto::WorkerSpec, mode: Mode) -> proto::RegisterWorkerRequest {
    proto::RegisterWorkerRequest {
        spec: Some(spec),
        mode: mode as i32,
        wait_ready: false,
    }
}

/// The row of an answer that carries one.
fn row_of(event: &proto::WorkerEvent) -> proto::WorkerRow {
    match event.event.clone() {
        Some(Event::Registered(row) | Event::Relabelled(row)) => Some(row),
        Some(Event::Replaced(replaced)) => replaced.new,
        _ => None,
    }
    .expect("an answer that carries a row")
}

/// The number of the life the door makes `writes` membership writes after
/// the one it numbered `first`, by the contract's sentence on `incarnation`:
/// within one run and one term the count moves with the store's writes.
fn later(first: u64, writes: u64) -> u64 {
    first + writes
}

/// The value of an unlabelled counter in a scrape.
fn counter(scrape: &str, name: &str) -> Option<u64> {
    scrape
        .lines()
        .find_map(|line| line.strip_prefix(name)?.strip_prefix(' ')?.parse().ok())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_leader_takes_a_fleet_through_the_door_and_polls_the_engine_it_was_given() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let mut replicas: Vec<Replica> = ["replica-a", "replica-b"]
        .iter()
        .map(|id| {
            let named = ["--identity-as-told", "control"].map(OsStr::new);
            Replica::start_with(id, Lease::File(&lease), settings(), 0, &named).expect("a start")
        })
        .collect();
    let mut record = Record::default();
    let (leader, term) = record.agreed(&mut replicas).await.expect("one leader");
    let follower = 1 - leader;
    let (engine, _engine_stop) = mock_engine().await;
    let phantom = |n: u8| format!("http://127.0.0.1:9/engine-{n}");

    // Each listener's word for the service's name: the leader's control
    // listener serves, the follower's does not, and no other listener
    // serves a call of its scopes yet.
    assert_eq!(
        word_within(
            replicas[leader].control,
            ServingStatus::Serving,
            Duration::from_secs(10)
        )
        .await,
        ServingStatus::Serving
    );
    assert_eq!(
        word(replicas[follower].control).await,
        ServingStatus::NotServing
    );
    for replica in &replicas {
        for addr in [replica.gateway, replica.replication] {
            assert_eq!(word(addr).await, ServingStatus::NotServing);
        }
    }

    // A follower takes no write and names the leader, as `Leader` does.
    let mut at_follower = client(replicas[follower].control).await;
    let refused = at_follower
        .register_worker(stated(ask(spec(&phantom(1)), Mode::Add)))
        .await
        .expect_err("a follower takes no write");
    assert_eq!(
        (refused.code(), refused.message()),
        (Code::FailedPrecondition, "this replica does not lead")
    );
    // The details are a `google.rpc.Status`, the form every gRPC client
    // decodes them as; the address is the leader's gateway listener.
    let named = replicas::not_leader(&refused);
    assert_eq!(named.leader_address, replicas[leader].gateway.to_string());

    // The leader takes three checked workers: each answer is the row as
    // stated, pending, at its own epoch, under the leader's term.
    let mut operator = client(replicas[leader].control).await;
    // A call that states no name never reaches the door.
    let unnamed = operator
        .register_worker(ask(spec(&phantom(1)), Mode::Add))
        .await
        .expect_err("no principal stands on the call");
    assert_eq!(unnamed.code(), Code::Unauthenticated);
    let mut rows = Vec::new();
    let mut counted_from = None;
    for n in 1..=3 {
        let told = operator
            .register_worker(stated(ask(spec(&phantom(n)), Mode::Add)))
            .await
            .expect("registered")
            .into_inner();
        assert!(matches!(told.event, Some(Event::Registered(_))), "{told:?}");
        assert_eq!((told.membership_epoch, told.term), (u64::from(n), term));
        let row = row_of(&told);
        // The count starts at the process's draw: the first answer says
        // where, under the leader's term.
        let first = *counted_from.get_or_insert(row.incarnation);
        assert_eq!(first >> 40, term);
        assert_eq!(
            (row.url.as_str(), row.incarnation, row.revision, row.status),
            (
                phantom(n).as_str(),
                later(first, u64::from(n) - 1),
                1,
                proto::WorkerStatus::Pending as i32
            )
        );
        rows.push(row);
    }

    // The engine, its checks off and waited for: ready at once, at the
    // revision its registration made (a status change moves none).
    let engine_url = format!("http://{engine}");
    // It states its bootstrap port, and the answer's row tells it back.
    let with_port = proto::WorkerSpec {
        pairing: Some(proto::PairingDescriptor {
            bootstrap_port: 8_998,
            ..Default::default()
        }),
        ..unchecked(&engine_url)
    };
    let waited = proto::RegisterWorkerRequest {
        wait_ready: true,
        ..ask(with_port, Mode::Add)
    };
    let told = operator
        .register_worker(stated(waited))
        .await
        .expect("registered")
        .into_inner();
    let ready = row_of(&told);
    assert_eq!(
        (ready.status, ready.revision, told.membership_epoch),
        (proto::WorkerStatus::Ready as i32, 1, 5)
    );
    assert_eq!(
        ready.pairing.as_ref().map(|pairing| pairing.bootstrap_port),
        Some(8_998)
    );

    // A second add is refused; an equal upsert writes nothing; a label is a
    // change in place; a replace is the next incarnation.
    let again = operator
        .register_worker(stated(ask(spec(&phantom(1)), Mode::Add)))
        .await
        .expect_err("the URL stands");
    assert_eq!(again.code(), Code::AlreadyExists);
    let same = operator
        .register_worker(stated(ask(spec(&phantom(1)), Mode::Upsert)))
        .await
        .expect("answered")
        .into_inner();
    assert_eq!((row_of(&same), same.membership_epoch), (rows[0].clone(), 5));
    let labelled = proto::WorkerSpec {
        labels: [("tier".to_owned(), "gold".to_owned())].into(),
        ..spec(&phantom(1))
    };
    let moved = operator
        .register_worker(stated(ask(labelled, Mode::Upsert)))
        .await
        .expect("relabelled")
        .into_inner();
    assert!(
        matches!(moved.event, Some(Event::Relabelled(_))),
        "{moved:?}"
    );
    assert_eq!(
        (
            row_of(&moved).revision,
            row_of(&moved).id,
            moved.membership_epoch
        ),
        (2, rows[0].id, 6)
    );
    let replaced = operator
        .register_worker(stated(ask(spec(&phantom(2)), Mode::Replace)))
        .await
        .expect("replaced")
        .into_inner();
    let Some(Event::Replaced(replaced)) = replaced.event else {
        panic!("a replace answers both rows: {replaced:?}");
    };
    assert_eq!(replaced.old, Some(rows[1].clone()));
    let new = replaced.new.expect("the new row");
    assert_eq!(
        (new.incarnation, new.revision),
        (later(rows[0].incarnation, 6), 1)
    );
    assert_ne!(new.id, rows[1].id);

    // A patch merges under the revision its caller read; a stale one is
    // refused.
    let patch = proto::PatchWorkerRequest {
        worker_id: rows[2].id,
        labels: [("lane".to_owned(), "b".to_owned())].into(),
        revision: rows[2].revision,
        ..Default::default()
    };
    let patched = operator
        .patch_worker(stated(patch.clone()))
        .await
        .expect("patched")
        .into_inner();
    assert_eq!(
        row_of(&patched).labels,
        [("lane".to_owned(), "b".to_owned())].into()
    );
    let stale = operator
        .patch_worker(stated(patch))
        .await
        .expect_err("a stale revision");
    assert_eq!(
        (stale.code(), stale.message()),
        (Code::FailedPrecondition, "the row is at revision 2")
    );

    // A registration the store cannot hold whole is refused by its path.
    let wide = proto::WorkerSpec {
        dp_size: 65,
        ..spec(&phantom(9))
    };
    let refused = operator
        .register_worker(stated(ask(wide, Mode::Add)))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().starts_with("spec.dp_size: "),
        "{}",
        refused.message()
    );

    // A URL that would make a row past a frame is refused by its field: no
    // call is taken and then answered in a frame its client cannot read.
    let long = spec(&format!("grpc://{}", "u".repeat(1_400_000)));
    let refused = operator
        .register_worker(stated(ask(long, Mode::Add)))
        .await
        .expect_err("refused");
    assert_eq!(refused.code(), Code::InvalidArgument);
    assert!(
        refused.message().starts_with("spec.url: "),
        "{}",
        refused.message()
    );

    // A drain answers the status change, a removal the row as it last stood.
    let drain = proto::DeregisterWorkerRequest {
        worker_id: rows[2].id,
        revision: 2,
        drain: true,
    };
    let draining = operator
        .deregister_worker(stated(drain))
        .await
        .expect("draining")
        .into_inner();
    assert_eq!(
        draining.event,
        Some(Event::Status(proto::StatusChanged {
            worker_id: rows[2].id,
            old: proto::WorkerStatus::Pending as i32,
            new: proto::WorkerStatus::Draining as i32,
            health: proto::HealthFlag::Draining as u32,
        }))
    );
    // The removal names the revision the drain was sent under: the drain
    // moved the status, not the stated facts.
    let removal = proto::DeregisterWorkerRequest {
        drain: false,
        ..drain
    };
    let removed = operator
        .deregister_worker(stated(removal))
        .await
        .expect("removed")
        .into_inner();
    assert_eq!(
        removed.event,
        Some(Event::Removed(proto::Removed {
            worker_id: rows[2].id,
            registration_key: phantom(3),
            revision: 2,
            draining: true,
        }))
    );

    // A removal that arrives twice (its first answer lost) with the URL's
    // next registration between names a registration that left: the new
    // one stands.
    let first = row_of(
        &operator
            .register_worker(stated(ask(spec(&phantom(4)), Mode::Add)))
            .await
            .expect("registered")
            .into_inner(),
    );
    let lost = proto::DeregisterWorkerRequest {
        worker_id: first.id,
        revision: first.revision,
        drain: false,
    };
    operator
        .deregister_worker(stated(lost))
        .await
        .expect("removed");
    let second = row_of(
        &operator
            .register_worker(stated(ask(spec(&phantom(4)), Mode::Add)))
            .await
            .expect("registered again")
            .into_inner(),
    );
    assert_eq!(second.revision, first.revision);
    assert_ne!(second.id, first.id);
    let retried = operator
        .deregister_worker(stated(lost))
        .await
        .expect_err("the retry names a registration that left");
    assert_eq!(retried.code(), Code::NotFound);

    // The leader's own load monitor polls the one ready engine it was given:
    // its counter moves within two of its poll periods. The follower was given
    // nothing and polls nothing.
    let started = Instant::now();
    let polled = loop {
        let (status, scrape) = http_get(replicas[leader].admin, "/metrics")
            .await
            .expect("the leader's scrape");
        assert_eq!(status, 200);
        let rows = counter(&scrape, "smg_policy_controller_load_rows_total");
        if rows.is_some_and(|rows| rows > 0) {
            break rows;
        }
        assert!(
            started.elapsed() < 2 * Config::DEFAULT_LOAD_POLL_INTERVAL + Duration::from_secs(20),
            "the leader polled no row of the engine it was given: {rows:?}"
        );
        tokio::time::sleep(Duration::from_millis(100)).await;
    };
    writeln!(
        std::io::stdout(),
        "the leader read {polled:?} load rows {:?} after the engine's registration",
        started.elapsed()
    )
    .expect("stdout");
    let (_, scrape) = http_get(replicas[follower].admin, "/metrics")
        .await
        .expect("the follower's scrape");
    assert_eq!(
        counter(&scrape, "smg_policy_controller_load_rows_total"),
        Some(0)
    );
    record.check().expect("the election's promise held");
}

fn listener(name: &str, scopes: &[Scope]) -> Listener {
    Listener {
        name: name.to_owned(),
        addr: loopback(),
        scopes: scopes.to_vec(),
        tls: None,
    }
}

fn config() -> Config {
    Config {
        controller_id: "door-0".to_owned(),
        listeners: vec![
            listener("gateway", &[Scope::Gateway, Scope::Ingestion]),
            listener("control", &[Scope::Control]),
            listener("replication", &[Scope::Replication]),
        ],
        admin: loopback(),
        shutdown_settle: Duration::from_millis(50),
        shutdown_grace: Duration::from_secs(5),
        load_poll_interval: Config::DEFAULT_LOAD_POLL_INTERVAL,
        decide_shards: 2,
        election: ElectionConfig::default(),
        discovery: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_store_holds_the_rows_the_door_answered() {
    let mut controller = Controller::bind(config()).await.expect("bind");
    for name in ["gateway", "control"] {
        controller
            .identity_as_told(name)
            .expect("a listener of the configuration");
    }
    let addresses = controller.addresses();
    let store = controller.store();
    let mut readiness = controller.readiness();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), readiness.wait_for(State::Leader)).await,
            Ok(Some(State::Leader)),
            "the lone replica leads within 30 s"
        );
        let control = addresses.of(Scope::Control).expect("the control listener");
        let gateways = addresses.of(Scope::Gateway).expect("the gateway listener");
        // The door's word is the control listener's: the gateways' listener
        // takes a principal too and serves no call of its scopes yet.
        assert_eq!(
            word_within(control, ServingStatus::Serving, Duration::from_secs(10)).await,
            ServingStatus::Serving
        );
        assert_eq!(word(gateways).await, ServingStatus::NotServing);
        let mut operator = client(control).await;

        // The registration calls are the control scope's: the gateways'
        // listener denies them.
        let mut gateway = client(gateways).await;
        let denied = gateway
            .register_worker(stated(ask(spec("http://127.0.0.1:9/a"), Mode::Add)))
            .await
            .expect_err("not a call of the gateway scope");
        assert_eq!(denied.code(), Code::PermissionDenied);
        assert_eq!(store.epoch(), 0);

        let checked = operator
            .register_worker(stated(ask(spec("http://127.0.0.1:9/a"), Mode::Add)))
            .await
            .expect("registered")
            .into_inner();
        let waited = proto::RegisterWorkerRequest {
            wait_ready: true,
            ..ask(unchecked("http://127.0.0.1:9/b"), Mode::Add)
        };
        let unchecked = operator
            .register_worker(stated(waited))
            .await
            .expect("registered")
            .into_inner();

        // The store's rows are the rows the answers told, in the snapshot's
        // order (by key).
        let (epoch, rows) = store.all_rows();
        assert_eq!(epoch, unchecked.membership_epoch);
        let told = [row_of(&checked), row_of(&unchecked)];
        assert_eq!(rows.len(), 2);
        for (stored, told) in rows.iter().zip(&told) {
            assert_eq!(
                (
                    stored.id.0,
                    stored.facts.key.as_str(),
                    stored.facts.incarnation,
                    stored.revision,
                    stored.facts.health.probes
                ),
                (
                    told.id,
                    told.url.as_str(),
                    told.incarnation,
                    told.revision,
                    !told.health_policy.is_some_and(|policy| policy.disabled)
                )
            );
        }
        assert_eq!(
            rows.iter().map(|row| row.status).collect::<Vec<_>>(),
            [WorkerStatus::Pending, WorkerStatus::Ready]
        );

        // A registration that states two ranks is answered a row a rank.
        let ranked = proto::WorkerSpec {
            dp_size: 2,
            ..spec("http://127.0.0.1:9/c")
        };
        let ranks = operator
            .register_worker(stated(ask(ranked, Mode::Add)))
            .await
            .expect("registered")
            .into_inner();
        let Some(Event::Ranks(answer)) = ranks.event else {
            panic!("no ranks in {ranks:?}");
        };
        assert_eq!(answer.rows.len(), 2);
        for rank in 0..2 {
            let url = format!("http://127.0.0.1:9/c@{rank}");
            assert!(store.by_key(&url).is_some(), "{url}");
        }

        // A removal through the door leaves the store without the row.
        let removal = proto::DeregisterWorkerRequest {
            worker_id: told[0].id,
            revision: told[0].revision,
            drain: false,
        };
        let removed = operator
            .deregister_worker(stated(removal))
            .await
            .expect("removed")
            .into_inner();
        assert_eq!(store.by_key("http://127.0.0.1:9/a"), None);
        assert_eq!(store.epoch(), removed.membership_epoch);

        stop.send(()).expect("the serve still runs");
    };
    let (served, ()) = Box::pin(tokio::time::timeout(Duration::from_secs(120), async {
        tokio::join!(serve, case)
    }))
    .await
    .expect("the case and the drain end within 120 s");
    served.expect("a clean drain");
}

/// A door that takes nothing serves nothing. On a control listener its
/// operator did not name, a leader refuses a registration whatever it
/// states, writes nothing, and keeps the listener's word for the service's
/// name at `NOT_SERVING`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_control_listener_its_operator_did_not_name_takes_no_registration() {
    let controller = Controller::bind(config()).await.expect("bind");
    let addresses = controller.addresses();
    let store = controller.store();
    let mut readiness = controller.readiness();
    let (stop, stopped) = oneshot::channel::<()>();
    let serve = controller.serve(async {
        let _ = stopped.await;
    });
    let case = async {
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(30), readiness.wait_for(State::Leader)).await,
            Ok(Some(State::Leader)),
            "the lone replica leads within 30 s"
        );
        let control = addresses.of(Scope::Control).expect("the control listener");
        let refused = client(control)
            .await
            .register_worker(stated(ask(spec("http://127.0.0.1:9/a"), Mode::Add)))
            .await
            .expect_err("the listener takes no identity as told");
        assert_eq!(refused.code(), Code::Unauthenticated);
        assert_eq!(store.epoch(), 0);
        // The replica leads and the word does not follow it.
        assert_eq!(
            word_within(control, ServingStatus::Serving, Duration::from_secs(1)).await,
            ServingStatus::NotServing
        );

        stop.send(()).expect("the serve still runs");
    };
    let (served, ()) = Box::pin(tokio::time::timeout(Duration::from_secs(120), async {
        tokio::join!(serve, case)
    }))
    .await
    .expect("the case and the drain end within 120 s");
    served.expect("a clean drain");
}
