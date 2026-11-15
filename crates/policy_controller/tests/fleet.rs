//! The end-to-end case, the harness first: a controller on ephemeral ports,
//! a fleet of mock engines and the generated client, in one process. The
//! fleet registers; the controller's own load driver polls it; a probe turn
//! of the case asks each engine its health and writes what it heard; and
//! the pool is read back as a decision's view reads it: every rank a row,
//! healthy by its probe, its load from a poll, an engine that holds
//! requests reading them. Then one engine is killed: the first dial that
//! hears it dead writes it, and the pool reads its rows unhealthy and every
//! other row healthy. Next, an engine stops answering with its sockets
//! open: a probe turn waits out its dial bound, writes it and goes on. Last,
//! the gateway sends requests of its own and reports each on its session:
//! the store holds the completion of the engine that answered, none of class
//! ok for the dead and the hung one, and nothing of it for another gateway;
//! a frame that reads wrong ends the session of the gateway that sent it.
//!
//! No decision is asked yet: the service answers `Leader` and takes the
//! session's reports, and the case asserts both end to end. The decisions
//! take the views' place when the service answers them.
//!
//! No stub server: the engines are the mock worker's, the controller is the
//! one the binary runs and the client is the generated one. Every wait is
//! bounded and named, and the case stands under one outer bound.

#[expect(
    dead_code,
    reason = "this binary's engines run on runtimes of their own; the engine served on the test's runtime is the load path's"
)]
#[path = "common/engines.rs"]
mod engines;
#[path = "common/fleet.rs"]
mod fleet;
#[expect(dead_code, reason = "this binary reads no admin page")]
#[path = "common/net.rs"]
mod net;

use std::{
    io::Write,
    sync::mpsc,
    time::{Duration, Instant},
};

use engines::cpu_seconds;
use fleet::{
    within, DialEnd, Harness, Replica, Shape, DIAL_TIMEOUT, GATEWAY, POLL_INTERVAL, PROBE_INTERVAL,
    ROOM,
};
use net::vm_rss_kib;
use smg_policy_controller::{
    client::{self, blocking, Client},
    controller::{State, PROTOCOL_VERSION},
    proto::{completed::Class, principal, released, Completed, Released, Side},
    store::{LaneCounts, WorkerStatus},
    view::{Role, SignalSource},
};
use tokio::sync::oneshot;
use tonic::Code;

const MODEL: &str = "fleet-model";

/// The fleet: four engines of one rank and one of two, one model.
const FLEET: [Shape; 5] = [
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 1,
    },
    Shape {
        model: MODEL,
        role: Role::Single,
        ranks: 2,
    },
];

/// The requests the case holds on one engine.
const HELD: u32 = 3;

/// The whole case's bound: its waits sum to seconds on an idle host.
const CASE: Duration = Duration::from_secs(180);

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_fleet_registers_its_loads_follow_and_a_dead_engine_leaves_the_healthy_set() {
    let case = async {
        let started = Instant::now();
        let cpu = cpu_seconds();
        let mut harness = Harness::start().await.expect("the harness starts");

        // The session. The leader welcomed the client's hello under the name
        // every call of the client states.
        let welcome = harness.welcome();
        assert_eq!(welcome.protocol, PROTOCOL_VERSION, "{welcome:?}");
        assert_eq!(welcome.controller_id, "fleet-controller");
        let told = welcome
            .principal
            .as_ref()
            .expect("a welcome names its session's principal");
        assert_eq!(
            (told.subject.as_str(), told.scope(), told.method()),
            (
                "fleet-gateway",
                principal::Scope::Gateway,
                principal::Method::AsTold
            ),
            "{told:?}"
        );
        assert!(
            welcome.grace_period_ms > 0 && welcome.leader_poll_ms > 0,
            "a welcome names its periods: {welcome:?}"
        );
        assert!(
            harness.term_floor() >= 1,
            "the client's floor stands at an answered term before it asks again"
        );

        // Ask: who leads, end to end.
        let leader = harness.leader().await.expect("the leader call");
        assert!(leader.is_leader, "a lone controller leads: {leader:?}");
        assert_eq!(leader.controller_id, "fleet-controller");
        assert!(
            leader.term >= 1,
            "a leader answers under a term: {leader:?}"
        );
        assert_eq!(
            harness.term_floor(),
            leader.term,
            "the client's floor is the leader's term"
        );

        // Register. Every rank is a row at once: none healthy before a
        // probe heard its engine, none with a load before a poll read one.
        let engines = harness.register(&FLEET).await.expect("the fleet registers");
        let (busy, dead) = (engines[1], engines[4]);
        let ranks: usize = FLEET.iter().map(|shape| usize::from(shape.ranks)).sum();
        let pool = harness.view(MODEL, Role::Single).expect("the view");
        assert_eq!(pool.len(), ranks, "every rank is a row: {pool:?}");
        assert!(
            pool.iter().all(|row| !row.healthy && row.load.is_none()),
            "nothing is known of an engine nobody asked: {pool:?}"
        );

        // Feed. The probe's verdict makes a rank ready and the controller's
        // own driver then polls its engine.
        harness.feed();
        let ready = within(
            "every rank is healthy and holds a polled load row",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.view(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter().all(|row| {
                        row.healthy
                            && row
                                .load
                                .is_some_and(|load| load.source == SignalSource::Poll)
                    })
                })
            },
        )
        .await
        .expect("the fleet turns ready");
        let pool = harness.view(MODEL, Role::Single).expect("the view");
        assert_eq!(pool.len(), ranks, "{pool:?}");
        for row in &pool {
            let load = row.load.expect("a polled row");
            assert!(
                load.max_running > 0,
                "an engine reports its admission window: {row:?}"
            );
            assert_eq!(
                (load.running, load.queued),
                (0, 0),
                "an idle fleet: {row:?}"
            );
        }
        let mut two: Vec<u16> = pool
            .iter()
            .filter(|row| row.engine == dead)
            .map(|row| row.rank)
            .collect();
        two.sort_unstable();
        assert_eq!(two, [0, 1], "each rank of one engine is a row of its own");

        // The loads follow the fleet: requests held on one engine reach its
        // row from that engine's own report, and no other row.
        harness
            .hold(busy, HELD as usize)
            .await
            .expect("the held requests");
        let followed = within(
            "the busy engine's row reads its held requests",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.view(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter().any(|row| {
                        row.engine == busy && row.load.is_some_and(|load| load.running == HELD)
                    })
                })
            },
        )
        .await
        .expect("the loads follow");
        let pool = harness.view(MODEL, Role::Single).expect("the view");
        for row in pool.iter().filter(|row| row.engine != busy) {
            assert_eq!(
                row.load.map(|load| load.running),
                Some(0),
                "only the busy engine runs requests: {row:?}"
            );
        }

        // The KV events follow the fleet: the busy engine's held requests
        // stored blocks, its stream's batches reached its rank's cursor, and
        // its row's index holds every block the engine holds.
        let stored = within(
            "the busy engine's blocks reach its row's index",
            PROBE_INTERVAL + ROOM,
            || {
                harness.heard(busy).first().is_some_and(|rank| {
                    rank.indexed > 0 && rank.cursor.is_some_and(|cursor| cursor.applied >= 1)
                })
            },
        )
        .await
        .expect("the KV events follow");
        let heard = harness.heard(busy);
        let rank = heard.first().expect("the busy engine's one rank");
        assert!(
            rank.last >= 1
                && rank.cursor.is_some_and(|cursor| {
                    cursor.applied >= rank.last && !cursor.replay_pending && !cursor.degraded
                }),
            "the cursor stands at what the stream delivered: {rank:?}"
        );
        assert!(
            rank.applied >= 1 && rank.indexed == rank.held,
            "the index holds each block the engine's events left it holding: {rank:?}"
        );
        assert_eq!(
            rank.counts,
            Some(LaneCounts::default()),
            "no store came without its parent and no block was refused: {rank:?}"
        );
        for engine in engines.iter().filter(|engine| **engine != busy) {
            for rank in harness.heard(*engine) {
                assert!(
                    rank.applied == 0 && rank.indexed == 0 && rank.held == 0,
                    "an engine that stored no block published none: engine {engine} {rank:?}"
                );
            }
        }
        assert_eq!(harness.gaps(), 0, "no stream named a gap");
        assert_eq!(
            harness.snapshot_chunks(),
            0,
            "an engine's own stream carries no snapshot chunk"
        );

        // The death. The kill waits for a view that reads its engine
        // healthy, so the death alone moves its rows. Its promise is held
        // by order below, whatever the host's clock does: the time from the
        // kill to the view that reads the death is a number of the case.
        within("the engine to kill reads healthy", ROOM, || {
            harness.view(MODEL, Role::Single).is_ok_and(|pool| {
                pool.iter()
                    .filter(|row| row.engine == dead)
                    .all(|row| row.healthy)
            })
        })
        .await
        .expect("a live engine reads healthy");
        let death = harness.kill(dead).await.expect("the kill");
        within(
            "a dead engine's rows read unhealthy and every other row healthy",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness
                    .view(MODEL, Role::Single)
                    .is_ok_and(|pool| pool.iter().all(|row| row.healthy == (row.engine != dead)))
            },
        )
        .await
        .expect("the death is seen");
        let seen = death.at.elapsed();
        let pool = harness.view(MODEL, Role::Single).expect("the view");
        assert_eq!(
            pool.len(),
            ranks,
            "a dead engine's rows stay rows: {pool:?}"
        );
        // The death's verdicts are the ones written from the kill on, one a
        // rank, by the first dial that heard the engine dead: a turn in
        // flight at the kill, or the first one begun after the death. The
        // liveness rule moves a not-ready row on to failed three failures
        // later: that step follows the death, it is not its verdict.
        let moved: Vec<_> = harness
            .verdicts()
            .into_iter()
            .filter(|verdict| {
                verdict.engine == dead
                    && verdict.at >= death.at
                    && verdict.to != WorkerStatus::Failed
            })
            .collect();
        assert_eq!(
            moved.len(),
            2,
            "one verdict a rank from the kill on: {moved:?}"
        );
        let first = harness
            .first_unhealthy(dead, death.asked)
            .expect("a dial heard the engine dead");
        assert!(
            first <= death.turn + 1,
            "the first turn after the death dialed it: turn {first} after {death:?}"
        );
        for verdict in &moved {
            assert_eq!(
                (verdict.from, verdict.to),
                (WorkerStatus::Ready, WorkerStatus::NotReady),
                "{verdict:?}"
            );
            assert_eq!(
                verdict.turn, first,
                "the first dial that heard the engine dead wrote it: {verdict:?} after {death:?}"
            );
        }
        let written = moved
            .iter()
            .map(|verdict| verdict.at.duration_since(death.at))
            .max()
            .unwrap_or_default();

        // The hang. An engine that stops answering with its sockets open
        // costs a probe turn its dial bound and no more: that turn writes
        // the engine's row not ready, and no live row moves.
        let hung = engines[2];
        let hang = Instant::now();
        harness.hang(hung).expect("the hang");
        within(
            "a hung engine's rows read unhealthy and every live row healthy",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness.view(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter()
                        .all(|row| row.healthy == (row.engine != dead && row.engine != hung))
                })
            },
        )
        .await
        .expect("the hang is seen");
        let hung_seen = hang.elapsed();
        let moved: Vec<_> = harness
            .verdicts()
            .into_iter()
            .filter(|verdict| {
                verdict.engine == hung && verdict.at >= hang && verdict.to != WorkerStatus::Failed
            })
            .collect();
        assert_eq!(
            moved
                .iter()
                .map(|verdict| (verdict.from, verdict.to))
                .collect::<Vec<_>>(),
            [(WorkerStatus::Ready, WorkerStatus::NotReady)],
            "one verdict for the hung engine's one rank: {moved:?}"
        );
        // That turn went on to every engine after the hung one, and from
        // the hang on no live row moved.
        let turn = moved[0].turn;
        within("the turn that waited out the bound ends", ROOM, || {
            harness.dialed(turn).len() == FLEET.len()
        })
        .await
        .expect("the turn goes on");
        assert_eq!(
            harness.dialed(turn),
            [
                (engines[0], DialEnd::Answered(true)),
                (busy, DialEnd::Answered(true)),
                (hung, DialEnd::TimedOut),
                (engines[3], DialEnd::Answered(true)),
                (dead, DialEnd::Failed),
            ],
            "the turn that waited out the bound went on to every engine"
        );
        let others: Vec<_> = harness
            .verdicts()
            .into_iter()
            .filter(|verdict| {
                verdict.at >= hang && verdict.engine != hung && verdict.engine != dead
            })
            .collect();
        assert!(
            others.is_empty(),
            "no live row moves at the hang: {others:?}"
        );
        // A hung engine is not a dead one: its listener still takes a
        // connection.
        harness
            .connect(hung)
            .await
            .expect("a hung engine takes a connection");
        // So the bound is spent on a fresh dial too: the next turn dials the
        // hung engine anew and waits it out again.
        within(
            "a hung engine is dialed again",
            2 * (PROBE_INTERVAL + DIAL_TIMEOUT) + ROOM,
            || harness.ends(hung, turn).len() >= 2,
        )
        .await
        .expect("the next turn dials the hung engine");
        assert_eq!(
            harness.ends(hung, turn)[..2],
            [DialEnd::TimedOut, DialEnd::TimedOut],
            "each dial of a hung engine waits out its bound"
        );

        // The reports. A gateway that decides by itself tells the controller
        // each request's life on its session, and the store holds what a
        // decision's precedence rule reads of it: one gateway's own
        // completions of class ok, a worker at a time.
        let live = engines[0];
        let asked = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers to the end");
        within(
            "the live engine's row holds this gateway's completion",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, asked),
        )
        .await
        .expect("the completion is read");
        let class = harness.send(dead).await.expect("a dead engine is sent");
        assert_eq!(class, Class::ConnectFailed, "a dead engine connects none");
        let class = tokio::time::timeout(ROOM, harness.send(hung))
            .await
            .expect("a hung engine's send ends at the dial bound")
            .expect("a hung engine is sent");
        assert_eq!(class, Class::Timeout, "a hung engine answers nothing");
        // A lease no leader issued is given up: not on file, no error, and
        // the session outlives it: the next request's reports are applied.
        harness
            .client()
            .released(Released {
                what: Some(released::What::LeaseId(u64::MAX)),
                reason: released::Reason::ClientGone.into(),
                ..Released::default()
            })
            .await
            .expect("a release is reported");
        let again = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers a second time");
        within(
            "the session outlives a release of a lease not on file",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, again),
        )
        .await
        .expect("the second completion is read");
        // One session's frames are applied in their order: the dead and the
        // hung engine's completions are on file by now, and neither is one
        // of class ok.
        assert!(
            !harness.completed_ok(GATEWAY, dead, asked),
            "a dead engine's row holds no completion of class ok"
        );
        assert!(
            !harness.completed_ok(GATEWAY, hung, asked),
            "a hung engine's row holds no completion of class ok"
        );
        // One gateway's completions are its own.
        assert!(
            !harness.completed_ok("another-gateway", live, asked),
            "another gateway reads none of this gateway's completions"
        );
        // A frame that reads wrong ends the session of the gateway that sent
        // it, by its field's name: a second gateway gives up neither a
        // lease nor a ticket.
        let mut other =
            Client::connect(&format!("http://{}", harness.gateway()), "another-gateway")
                .await
                .expect("a second gateway connects");
        other
            .released(Released::default())
            .await
            .expect("the frame is queued");
        let ended = other
            .session_end()
            .await
            .expect("the controller ends the session inside the client's bound");
        assert_eq!(ended.code(), Code::InvalidArgument, "{ended:?}");
        assert!(
            ended.message().contains("Released.what"),
            "the refusal names the field: {ended:?}"
        );
        // The first gateway's session stands: its next reports are applied.
        let last = Instant::now();
        let class = harness.send(live).await.expect("a live engine is sent");
        assert_eq!(class, Class::Ok, "a live engine answers a third time");
        within(
            "the first gateway's session stands",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok(GATEWAY, live, last),
        )
        .await
        .expect("the third completion is read");

        // The blocking form's report leaves when it is made: its runtime
        // drives the connection between calls. The client is held, and
        // asked nothing more, until the store has read its completion.
        let served = harness.worker(live).expect("the live engine's row").0;
        let address = format!("http://{}", harness.gateway());
        let reported = Instant::now();
        let (hold, held) = mpsc::channel::<()>();
        let (told, reporting) = oneshot::channel();
        std::thread::spawn(move || {
            let report =
                blocking::Client::connect(&address, "blocking-gateway").and_then(|mut client| {
                    client.completed(Completed {
                        lease_id: 1,
                        side: Side::Single.into(),
                        served,
                        class: Class::Ok.into(),
                        ..Completed::default()
                    })?;
                    Ok(client)
                });
            match report {
                Ok(client) => {
                    let _ = told.send(Ok(()));
                    let _ = held.recv();
                    drop(client);
                }
                Err(error) => {
                    let _ = told.send(Err(error.to_string()));
                }
            }
        });
        tokio::time::timeout(ROOM, reporting)
            .await
            .expect("the blocking client reports inside its bound")
            .expect("the client's thread sends its answer")
            .expect("the blocking client's report");
        within(
            "the blocking form's report leaves when it is made",
            PROBE_INTERVAL + ROOM,
            || harness.completed_ok("blocking-gateway", live, reported),
        )
        .await
        .expect("the blocking gateway's completion is read");
        drop(hold);

        // The Numbers: one process holds the controller, the engines and
        // the feeds, so its CPU and resident memory are a ceiling over the
        // service's, never the service's own.
        let wall = started.elapsed();
        let mut out = std::io::stdout();
        writeln!(out, "FLEET_ENGINES {} FLEET_RANKS {ranks}", FLEET.len()).expect("stdout");
        writeln!(
            out,
            "FLEET_PROBE_INTERVAL_MS {}",
            PROBE_INTERVAL.as_millis()
        )
        .expect("stdout");
        writeln!(out, "FLEET_DIAL_TIMEOUT_MS {}", DIAL_TIMEOUT.as_millis()).expect("stdout");
        writeln!(out, "FLEET_READY_MS {}", ready.as_millis()).expect("stdout");
        writeln!(out, "FLEET_LOADS_FOLLOW_MS {}", followed.as_millis()).expect("stdout");
        writeln!(out, "FLEET_KV_FOLLOW_MS {}", stored.as_millis()).expect("stdout");
        writeln!(out, "FLEET_KV_BATCHES {}", rank.applied).expect("stdout");
        writeln!(out, "FLEET_KV_BLOCKS {}", rank.indexed).expect("stdout");
        writeln!(out, "FLEET_DEATH_WRITTEN_MS {}", written.as_millis()).expect("stdout");
        writeln!(out, "FLEET_DEATH_SEEN_MS {}", seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_HANG_SEEN_MS {}", hung_seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_WALL_MS {}", wall.as_millis()).expect("stdout");
        writeln!(out, "FLEET_CPU_MS {:.0}", (cpu_seconds() - cpu) * 1_000.0).expect("stdout");
        writeln!(out, "FLEET_RSS_KIB {}", vm_rss_kib()).expect("stdout");

        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
        .await
        .expect("the case ends inside its outer bound");
}

/// The blocking form of the reference client asks the same started
/// controller from a thread that runs no runtime, and holds the same floor.
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
                    Ok((leader, client.term_floor(), subject))
                });
            let _ = answered.send(asked.map_err(|error| error.to_string()));
        });
        let (leader, floor, subject) = tokio::time::timeout(ROOM, answer)
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
        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, case)
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
