//! The fleet's failover end to end: two controllers as processes of their
//! own over one lease file, mock engines, and the reference client as the
//! gateway. The standby follows its leader's log. The leader is killed
//! between two asks of the gateway's traffic, its lease unreleased; the
//! standby takes the lease when the dead leader's hold of it runs out, and
//! the gateway that turns to it is answered under a higher term, each
//! routing key on the worker the dead leader named, by a controller that
//! polls the engines itself.
//!
//! What crosses the take is what a standby applies: the fleet's rows, under
//! the ids the leader gave them, their loads, and the leader's document.
//! Each replica reads a spec file of its own, and both files change once
//! the fleet has registered, each to another document. The standby follows
//! its leader's and looks at no file while it follows. Promoted, it reads
//! its own file as a leader does, and applies it as the generation after
//! its leader's. No session's pin and no lease crosses a take in this build,
//! so the case asks keys the ring places and gives each lease up as it
//! goes.
//!
//! No call reads a following standby's rows from outside its process. The
//! traffic runs until the standby reads synced and has applied, on two
//! readings a poll apart, a frame younger than the fleet's last
//! registration: frames are applied in the log's order, so the fleet's rows
//! stand behind them. The standby listens at a second loopback address,
//! which a Linux host answers with no alias made: the binary is Linux's.
//! The pair's replication port is one of the pairs' own numbers
//! ([`pair::ports`]): the leader's own bind takes it, where nothing holds
//! it on the standby's address ([`pair::free`]).
#![cfg(target_os = "linux")]
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "this binary serves mock engines and waits by its own bound; the process's CPU time is the load path's"
)]
#[path = "common/engines.rs"]
mod engines;
#[expect(dead_code, reason = "this binary reads no resident size")]
#[path = "common/net.rs"]
mod net;
#[path = "common/pair.rs"]
mod pair;
#[expect(
    dead_code,
    reason = "this binary starts a pair over a lease file and kills its leader; the election's record and the other starts are the replicas cases'"
)]
#[path = "common/replicas.rs"]
mod replicas;

use std::{
    collections::{HashMap, HashSet},
    ffi::OsStr,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::Path,
    time::{Duration, Instant},
};

use engines::{eventually, mock_engine};
use openai_protocol::worker::RuntimeType;
use replicas::{Lease, Replica, ROOM};
use smg_policy_controller::service::{
    client::{self, Answer, Client},
    elector::ElectionSettings,
    proto::{
        self, decision::Outcome, identity::Form, policy_controller_client::PolicyControllerClient,
        register_worker_request::Mode, released, worker_event::Event, DecideRequest, Decision,
        Identity, Released, RequestFacts,
    },
};
use tonic::{metadata::AsciiMetadataValue, transport::Channel, Code, Request};

/// The one model the fleet serves.
const MODEL: &str = "fleet-model";

/// The name the gateway states.
const GATEWAY: &str = "fleet-gateway";

/// The name the operator states on the control listener.
const OPERATOR: &str = "fleet-operator";

/// The fleet's engines.
const ENGINES: usize = 3;

/// The routing keys the gateway asks, round after round.
const KEYS: u64 = 64;

/// The leader's address and the standby's: replicas of one lease listen on
/// one replication port, each at an address of its own.
const LEADER: IpAddr = IpAddr::V4(Ipv4Addr::LOCALHOST);
const STANDBY: IpAddr = IpAddr::V4(Ipv4Addr::new(127, 0, 0, 2));

/// How often each replica's load monitor polls an engine: the flag's one
/// second.
const POLL: Duration = Duration::from_secs(1);

/// A wait's step.
const PACE: Duration = Duration::from_millis(25);

/// What a standby logs when its stream to its leader stands.
const FOLLOWING: &str = "following the leader's log";

/// A standby's two series: whether it is synced with its leader, and the
/// age in seconds of the last frame it applied.
const SYNCED: &str = "smg_policy_controller_replication_synced";
const LAG: &str = "smg_policy_controller_replication_lag_seconds";

/// The document both replicas run: the ring preset's one profile.
const DOCUMENT: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [consistent-hash]\n    pick: best\n";

/// The leader's file after its change: the same profile under a written
/// binding, so the routing is the same and the document is another one.
const CHANGED: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [consistent-hash]\n    pick: best\nbindings:\n  - match: { model: \"*\" }\n    profile: default\n";

/// The standby's file after its change, a third document with the same
/// routing: what a promoted standby applies from its own file.
const OWN: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [consistent-hash]\n    pick: best\nbindings:\n  - match: { model: fleet-model }\n    profile: default\n  - match: { model: \"*\" }\n    profile: default\n";

/// `message` as a call that states the operator's name: a listener its
/// operator named takes the name a call states.
fn stating<T>(message: T) -> Request<T> {
    let mut call = Request::new(message);
    call.metadata_mut()
        .insert("x-smg-principal", AsciiMetadataValue::from_static(OPERATOR));
    call
}

/// Registers the engine at `engine` through the door as a worker of the
/// model and returns the worker's id. Its checks are off, so it is ready
/// from its registration: no prober of the case writes a store it does not
/// hold. The door opens once the election's steps are told, so a refusal is
/// asked again inside the room.
async fn through_the_door(
    door: &mut PolicyControllerClient<Channel>,
    engine: SocketAddr,
) -> Result<u64, String> {
    let ask = proto::RegisterWorkerRequest {
        spec: Some(proto::WorkerSpec {
            url: format!("http://{engine}"),
            wire: proto::WireMode::Grpc.into(),
            role: proto::Role::Single.into(),
            runtime: RuntimeType::Vllm.as_str().to_owned(),
            models: vec![MODEL.to_owned()],
            health_policy: Some(proto::HealthPolicy {
                disabled: true,
                ..proto::HealthPolicy::default()
            }),
            ..proto::WorkerSpec::default()
        }),
        mode: Mode::Add.into(),
        wait_ready: false,
    };
    let asked = Instant::now();
    loop {
        match door.register_worker(stating(ask.clone())).await {
            Ok(event) => {
                return match event.into_inner().event {
                    Some(Event::Registered(row)) => Ok(row.id),
                    other => Err(format!("the door answered a registration with {other:?}")),
                };
            }
            Err(_) if asked.elapsed() < ROOM => tokio::time::sleep(PACE).await,
            Err(status) => return Err(format!("the door took no registration: {status}")),
        }
    }
}

/// [`document`], asked again inside the room while the replica holds none: a
/// new leader's first document lands a moment behind the start of its lead,
/// and its standby is told it behind that.
async fn first_document(
    door: &mut PolicyControllerClient<Channel>,
) -> Result<proto::PolicySpecDocument, String> {
    let asked = Instant::now();
    loop {
        match document(door).await {
            Err(_) if asked.elapsed() < ROOM => tokio::time::sleep(PACE).await,
            held => return held,
        }
    }
}

/// The document the replica behind `door` runs, as its operator reads it.
async fn document(
    door: &mut PolicyControllerClient<Channel>,
) -> Result<proto::PolicySpecDocument, String> {
    door.get_policy_spec(stating(proto::GetPolicySpecRequest::default()))
        .await
        .map(tonic::Response::into_inner)
        .map_err(|status| format!("the door answered no document: {status}"))
}

/// The gateway's session at `replica`, opened once the replica leads: a
/// dial that is refused, or told another leader, is made again inside the
/// room.
async fn session_at(replica: &Replica) -> Result<Client, String> {
    let asked = Instant::now();
    loop {
        match Client::connect(&format!("http://{}", replica.gateway), GATEWAY).await {
            Ok(client) => return Ok(client),
            Err(_) if asked.elapsed() < ROOM => tokio::time::sleep(PACE).await,
            Err(refused) => {
                return Err(format!(
                    "{} does not lead inside the room: {refused}",
                    replica.id
                ))
            }
        }
    }
}

/// Whether `error` is a session its controller ended because it stopped
/// leading, or leads under a later term: a gateway opens its session again.
fn ended(error: &client::Error) -> bool {
    matches!(error, client::Error::Status(status) if status.code() == Code::Aborted)
}

/// One ask of the routing key `key`, the `n`th of the case: the worker its
/// pick names first. Its lease is given up on the session, as a gateway
/// whose client left. The session stands at `replica`.
///
/// A leader whose renewal lands past its renew deadline leads nobody until
/// one lands in time, and the lease's store on a busy host can hold a
/// renewal that long. Meanwhile it answers that it does not lead: a gateway
/// asks again. A renewal that took the deadline whole ends the session,
/// which a gateway hears at its next report: it opens its session again
/// and says the report there. The case does both, inside the room.
async fn placed(gateway: &mut Client, replica: &Replica, key: u64, n: u64) -> Result<u64, String> {
    let ask = DecideRequest {
        request_id: format!("ask-{n}").into_bytes(),
        facts: Some(RequestFacts {
            model: MODEL.to_owned(),
            seed: n,
            budget_micros: 5_000_000,
            session_key: Some(Identity {
                form: Some(Form::Hash(key)),
            }),
            ..RequestFacts::default()
        }),
        constraints: None,
        again: false,
        ..Default::default()
    };
    let began = Instant::now();
    let answer = loop {
        match gateway.decide(ask.clone()).await {
            Ok(Answer::Busy(busy))
                if busy.reason() == proto::busy::Reason::NotLeader && began.elapsed() < ROOM =>
            {
                tokio::time::sleep(PACE).await;
            }
            answer => break answer.map_err(|error| format!("no answer: {error}"))?,
        }
    };
    let Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    }) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    let worker = pick.targets.first().ok_or("a pick with no target")?.id;
    for lease in leases {
        let released = Released {
            what: Some(released::What::LeaseId(lease.id)),
            reason: released::Reason::ClientGone.into(),
            ..Released::default()
        };
        let reported = match gateway.released(released).await {
            Err(error) if ended(&error) => {
                *gateway = session_at(replica).await?;
                gateway.released(released).await
            }
            reported => reported,
        };
        reported.map_err(|error| format!("a lease is not given up: {error}"))?;
    }
    Ok(worker)
}

/// A series' value on `replica`'s metrics page, by its name: the last field
/// of the series' line.
async fn series(replica: &Replica, name: &str) -> Result<f64, String> {
    let (status, page) = net::http_get(replica.admin, "/metrics").await?;
    if status != 200 {
        return Err(format!("the metrics page answered {status}"));
    }
    page.lines()
        .find(|line| {
            line.strip_prefix(name)
                .is_some_and(|rest| rest.starts_with([' ', '{']))
        })
        .and_then(|line| line.rsplit(' ').next())
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| format!("no series {name} on the metrics page"))
}

/// Whether `standby` reads synced with its leader and applied its last
/// frame after `since`.
async fn applied_since(standby: &Replica, since: Instant) -> Result<bool, String> {
    let synced = series(standby, SYNCED).await? > 0.5;
    let age = Duration::try_from_secs_f64(series(standby, LAG).await?)
        .map_err(|error| format!("the lag is no age: {error}"))?;
    Ok(synced && age < since.elapsed())
}

/// The load polls the engine at `engine` has answered.
fn polls(engine: SocketAddr) -> usize {
    mock_worker::overrides::for_port(engine.port()).loads_polls()
}

/// A replica's flags: the spec file, the names its control and replication
/// listeners take as told, and a load poll a second.
fn flags(spec: &Path) -> [&OsStr; 6] {
    [
        OsStr::new("--spec-file"),
        spec.as_os_str(),
        OsStr::new("--identity-as-told"),
        OsStr::new("control,replication"),
        OsStr::new("--load-monitor-interval"),
        OsStr::new("1"),
    ]
}

/// How many of the pairs' own numbers a leader's start tries.
const NUMBERS: usize = 16;

/// The first replica of a pair, with the pair's replication port: under the
/// election's default timings, on a port of the pairs' own numbers, so no
/// bind of port 0 is handed it on either address. The leader's own bind
/// takes the number. A start that a holder of the number refuses goes on to
/// the next, and so does one whose number is held on the standby's address,
/// by a standby that outlived its leader.
fn first_of_the_pair(lease: &Path, flags: &[&OsStr; 6]) -> (Replica, u16) {
    let mut started = Err("no number was tried".to_owned());
    for port in pair::ports(LEADER).take(NUMBERS) {
        started = Replica::start_on(
            "fleet-leader",
            (LEADER, port),
            Lease::File(lease),
            ElectionSettings::default(),
            flags,
        )
        .and_then(|leader| {
            if pair::free(STANDBY, port).expect("the look's bind") {
                Ok((leader, port))
            } else {
                Err(format!("{port} is held on the standby's address"))
            }
        });
        if started.is_ok() {
            break;
        }
    }
    started.expect("the leader starts on a number of the pairs' own")
}

/// The whole case's bound: its waits sum to seconds on an idle host.
const CASE: Duration = Duration::from_secs(180);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_killed_leaders_standby_answers_each_routing_key_on_its_worker() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let spec = dir.path().join("policy.yaml");
        std::fs::write(&spec, DOCUMENT).expect("the leader's file");
        let flags = flags(&spec);
        let own = dir.path().join("standby.yaml");
        std::fs::write(&own, DOCUMENT).expect("the standby's file");
        let following = self::flags(&own);

        // The first replica leads alone; the second starts at an address of
        // its own on the same port, finds the lease held and follows the
        // leader's log.
        let (mut leader, port) = first_of_the_pair(&lease, &flags);
        let mut gateway = session_at(&leader).await.expect("the first replica leads");
        let mut standby = Replica::start_on(
            "fleet-standby",
            (STANDBY, port),
            Lease::File(&lease),
            ElectionSettings::default(),
            &following,
        )
        .expect("the standby starts");
        assert!(
            standby.logs(|replica| replica.logged(FOLLOWING)).await,
            "the standby follows its leader's log: {}",
            standby.log()
        );

        // The fleet registers at the leader, each engine ready from its
        // registration.
        let mut engines = Vec::with_capacity(ENGINES);
        let mut served = Vec::with_capacity(ENGINES);
        for _ in 0..ENGINES {
            let (engine, stop) = mock_engine(1).await.expect("a mock engine");
            engines.push(engine);
            served.push(stop);
        }
        let channel = Channel::from_shared(format!("http://{}", leader.control))
            .expect("a uri")
            .connect()
            .await
            .expect("the control listener accepts");
        let mut door = PolicyControllerClient::new(channel);
        let mut workers = Vec::with_capacity(ENGINES);
        for engine in &engines {
            let worker = through_the_door(&mut door, *engine)
                .await
                .expect("the door takes a worker");
            workers.push(worker);
        }

        // Both files change, each to another document. The leader's watch
        // applies its own as the next generation. The standby's watch shows
        // its door the file and applies nothing while it follows: its
        // document is its leader's.
        std::fs::write(&own, OWN).expect("the standby's file");
        std::fs::write(&spec, CHANGED).expect("the leader's file");
        let changed = Instant::now();
        loop {
            let held = first_document(&mut door)
                .await
                .expect("the leader answers its document");
            if held.generation == 2 {
                assert_eq!(held.document, CHANGED.as_bytes());
                break;
            }
            assert!(
                changed.elapsed() < ROOM,
                "the leader applies its changed file inside the room: it runs generation {}",
                held.generation
            );
            tokio::time::sleep(PACE).await;
        }
        // The standby's frames are read behind the registrations and the
        // leader's apply: both stand behind a frame younger than this.
        let registered = Instant::now();

        // A registration is in force for a decision within one publication
        // window of its answer. The keys are placed once a round of them
        // names every engine: each answer behind it is over the whole fleet.
        let mut asked = 0;
        let whole = Instant::now();
        loop {
            let mut named = HashSet::new();
            for key in 1..=KEYS {
                asked += 1;
                if let Ok(worker) = placed(&mut gateway, &leader, key, asked).await {
                    named.insert(worker);
                }
                // An ask can wait a room out by itself: the round ends at
                // its own room, and the assert below reads it.
                if whole.elapsed() >= ROOM {
                    break;
                }
            }
            if workers.iter().all(|worker| named.contains(worker)) {
                break;
            }
            assert!(
                whole.elapsed() < ROOM,
                "a round of the keys names every engine of the fleet inside the room: {named:?}"
            );
            tokio::time::sleep(PACE).await;
        }

        // The traffic. The gateway asks its keys round after round; the
        // first round places each key, and every later answer names the
        // key's worker. The rounds go on until the standby holds the fleet:
        // synced, with a frame younger than the last registration on two
        // readings a poll apart.
        let mut homes = HashMap::new();
        let mut read = Vec::new();
        while read.len() < 2 {
            for key in 1..=KEYS {
                asked += 1;
                let worker = placed(&mut gateway, &leader, key, asked)
                    .await
                    .expect("the leader answers a key over its fleet");
                assert!(
                    workers.contains(&worker),
                    "a pick names a worker of the fleet"
                );
                assert_eq!(
                    *homes.entry(key).or_insert(worker),
                    worker,
                    "a routing key returns to its worker"
                );
            }
            let since = read.last().copied().unwrap_or(registered);
            if since.elapsed() >= POLL
                && applied_since(&standby, since)
                    .await
                    .expect("the standby's metrics page")
            {
                read.push(Instant::now());
            }
            assert!(
                registered.elapsed() < ROOM,
                "the standby reads synced behind the fleet's registration inside the room"
            );
        }
        let places: HashSet<_> = homes.values().collect();
        assert!(
            places.len() > 1,
            "the ring spreads the keys over the fleet: {places:?}"
        );

        let led = gateway.term_floor();

        // The kill, between two asks of the traffic: the leader's process
        // is gone with its lease unreleased, and the ask that follows is
        // answered by nobody.
        leader.kill().expect("the kill");
        leader.exited().expect("a killed leader is gone");
        asked += 1;
        let unanswered = placed(&mut gateway, &leader, 1, asked).await;
        assert!(
            unanswered.is_err(),
            "a dead leader answers no ask: {unanswered:?}"
        );

        // The standby takes the lease when the dead leader's hold of it
        // runs out, and the gateway that turns to it is welcomed there.
        let mut gateway = session_at(&standby)
            .await
            .expect("the standby leads once the dead leader's hold runs out");
        assert_eq!(gateway.welcome().controller_id, "fleet-standby");
        assert!(
            gateway.term_floor() > led,
            "a promoted standby answers under a term above its leader's"
        );

        // Each routing key is on the worker the dead leader named, by the
        // id the leader gave the row: the ask the kill left unanswered
        // first, asked again under its own id.
        let worker = placed(&mut gateway, &standby, 1, asked)
            .await
            .expect("the promoted standby answers the ask the kill left unanswered");
        assert_eq!(
            worker, homes[&1],
            "a routing key keeps its worker when its leader is killed"
        );
        for key in 1..=KEYS {
            asked += 1;
            let worker = placed(&mut gateway, &standby, key, asked)
                .await
                .expect("the promoted standby answers a key over its leader's fleet");
            assert_eq!(
                worker, homes[&key],
                "a routing key keeps its worker when its leader is killed"
            );
        }

        // The promoted standby runs the document its leader applied, under
        // the leader's number, until its own watch reads its file as a
        // leader's does: the file changed while it followed, and it is
        // applied as the generation after the leader's. A standby that had
        // not followed the leader's document would count one less, and one
        // that had read its own file while it followed would not be here.
        let channel = Channel::from_shared(format!("http://{}", standby.control))
            .expect("a uri")
            .connect()
            .await
            .expect("the promoted standby's control listener accepts");
        let mut door = PolicyControllerClient::new(channel);
        let promoted = Instant::now();
        loop {
            let held = document(&mut door)
                .await
                .expect("the promoted standby answers its document");
            if held.generation == 3 {
                assert_eq!(held.document, OWN.as_bytes());
                break;
            }
            assert_eq!(
                (held.generation, held.document.as_slice()),
                (2, CHANGED.as_bytes()),
                "ahead of its own file's turn a promoted standby runs the document its leader applied"
            );
            assert!(
                promoted.elapsed() < ROOM,
                "a promoted standby reads its own file inside the room"
            );
            tokio::time::sleep(PACE).await;
        }

        // The promoted standby polls the engines itself: no other
        // controller is left to.
        let before: Vec<usize> = engines.iter().map(|engine| polls(*engine)).collect();
        let polled = eventually(ROOM, || {
            engines
                .iter()
                .zip(&before)
                .all(|(engine, before)| polls(*engine) >= before + 2)
        })
        .await;
        assert!(
            polled,
            "a promoted standby's own load monitor polls every engine"
        );

        standby.terminate().expect("the stop's signal");
        let stopped = standby.exited().expect("the promoted standby stops");
        assert!(stopped.success(), "a clean stop: {stopped}");
        drop(served);
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A rollout of a new file. A replica that starts over the new document
/// beside a leader that runs the old one follows it: it applies none of its
/// own at its start, takes its leader's first generation, reads synced and
/// answers its leader's document. A replica that ran its own document from
/// its start would hold the leader's number over other bytes, and follow
/// nothing. The old leader is killed: the new one runs what its file says,
/// the new document as the next generation, though its file never moved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_that_starts_over_another_document_follows_its_leader() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let spec = dir.path().join("policy.yaml");
        std::fs::write(&spec, DOCUMENT).expect("the leader's file");
        let own = dir.path().join("standby.yaml");
        std::fs::write(&own, OWN).expect("the standby's file");
        let (mut leader, port) = first_of_the_pair(&lease, &flags(&spec));
        let gateway = session_at(&leader).await.expect("the first replica leads");
        let mut standby = Replica::start_on(
            "fleet-standby",
            (STANDBY, port),
            Lease::File(&lease),
            ElectionSettings::default(),
            &flags(&own),
        )
        .expect("the standby starts");
        assert!(
            standby.logs(|replica| replica.logged(FOLLOWING)).await,
            "the standby follows its leader's log: {}",
            standby.log()
        );

        // Synced, with a frame applied behind its start, on two readings a
        // poll apart: the leader's first generation landed, and its frames
        // go on landing.
        let started = Instant::now();
        let mut read = Vec::new();
        while read.len() < 2 {
            let since = read.last().copied().unwrap_or(started);
            if since.elapsed() >= POLL
                && applied_since(&standby, since)
                    .await
                    .expect("the standby's metrics page")
            {
                read.push(Instant::now());
            }
            assert!(
                started.elapsed() < ROOM,
                "a standby that read another document at its start follows its leader inside the room: {}",
                standby.log()
            );
            tokio::time::sleep(PACE).await;
        }

        // What it runs is its leader's document, under the leader's number:
        // its own file's is applied nowhere while it follows.
        let channel = Channel::from_shared(format!("http://{}", standby.control))
            .expect("a uri")
            .connect()
            .await
            .expect("the standby's control listener accepts");
        let mut door = PolicyControllerClient::new(channel);
        let held = first_document(&mut door)
            .await
            .expect("the standby answers its document");
        assert_eq!(
            (held.generation, held.document.as_slice()),
            (1, DOCUMENT.as_bytes()),
            "a standby runs its leader's document"
        );

        // The old leader ends. The new one reads its file as a start does:
        // its own document is the next generation.
        drop(gateway);
        leader.kill().expect("the kill");
        leader.exited().expect("a killed leader is gone");
        let killed = Instant::now();
        loop {
            // A read is answered through the take: the standby's door holds
            // its document whoever leads.
            let held = document(&mut door)
                .await
                .expect("the standby answers its document");
            if held.generation == 2 {
                assert_eq!(held.document, OWN.as_bytes());
                break;
            }
            assert_eq!(
                (held.generation, held.document.as_slice()),
                (1, DOCUMENT.as_bytes()),
                "ahead of its own file's turn a promoted standby runs the document its leader applied"
            );
            assert!(
                killed.elapsed() < ROOM,
                "a new leader runs what its file says inside the room: {}",
                standby.log()
            );
            tokio::time::sleep(PACE).await;
        }

        standby.terminate().expect("the stop's signal");
        let stopped = standby.exited().expect("the standby stops");
        assert!(stopped.success(), "a clean stop: {stopped}");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}

/// A mount one step behind. Both replicas start over one document. The
/// standby's mounted file takes the first of two changes, and its watch
/// shows it to the door. The leader's file takes both, and the standby
/// follows each. The leader is killed. The new leader keeps the document
/// it followed: its file's document stood on it since the file took it,
/// and its leader replaced it, so the take brings nothing older back. The
/// file is left for its next move.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_promoted_standby_whose_file_lags_keeps_the_document_it_followed() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let spec = dir.path().join("policy.yaml");
        std::fs::write(&spec, DOCUMENT).expect("the leader's file");
        let own = dir.path().join("standby.yaml");
        std::fs::write(&own, DOCUMENT).expect("the standby's file");
        let (mut leader, port) = first_of_the_pair(&lease, &flags(&spec));
        let gateway = session_at(&leader).await.expect("the first replica leads");
        let mut standby = Replica::start_on(
            "fleet-standby",
            (STANDBY, port),
            Lease::File(&lease),
            ElectionSettings::default(),
            &flags(&own),
        )
        .expect("the standby starts");
        assert!(
            standby.logs(|replica| replica.logged(FOLLOWING)).await,
            "the standby follows its leader's log: {}",
            standby.log()
        );
        let channel = Channel::from_shared(format!("http://{}", standby.control))
            .expect("a uri")
            .connect()
            .await
            .expect("the standby's control listener accepts");
        let mut door = PolicyControllerClient::new(channel);

        // The first change reaches the standby's mount, and three looks of
        // its watch show it to the door. Then the leader's file takes both
        // changes, and the standby follows each: its own mount stays one
        // step behind.
        std::fs::write(&own, CHANGED).expect("the standby's file");
        tokio::time::sleep(3 * POLL).await;
        for (generation, written) in [(2, CHANGED), (3, OWN)] {
            std::fs::write(&spec, written).expect("the leader's file");
            let changed = Instant::now();
            loop {
                let held = document(&mut door)
                    .await
                    .expect("the standby answers its document");
                if held.generation == generation {
                    assert_eq!(held.document, written.as_bytes());
                    break;
                }
                assert!(
                    changed.elapsed() < ROOM,
                    "the standby follows its leader's changed document inside the room: {}",
                    standby.log()
                );
                tokio::time::sleep(PACE).await;
            }
        }

        // The leader ends, and the standby leads: three looks of its watch
        // later it runs the document it followed, under the same number.
        drop(gateway);
        leader.kill().expect("the kill");
        leader.exited().expect("a killed leader is gone");
        let gateway = session_at(&standby)
            .await
            .expect("the standby leads inside the room");
        tokio::time::sleep(3 * POLL).await;
        let held = document(&mut door)
            .await
            .expect("the new leader answers its document");
        assert_eq!(
            (held.generation, held.document.as_slice()),
            (3, OWN.as_bytes()),
            "a mount one step behind brings no older document back at a take"
        );

        drop(gateway);
        standby.terminate().expect("the stop's signal");
        let stopped = standby.exited().expect("the standby stops");
        assert!(stopped.success(), "a clean stop: {stopped}");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}
