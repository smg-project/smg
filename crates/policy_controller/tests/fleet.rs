//! The end-to-end case, the harness first: a controller on ephemeral ports,
//! a fleet of mock engines and the generated client, in one process. The
//! fleet registers, an engine of one rank through the control scope's call;
//! the controller's own load driver polls it; a probe turn of the case asks
//! each engine its health and writes what it heard; and the pool is read
//! back as a decision's view reads it: every rank a row, healthy by its
//! probe, its load from a poll, an engine that holds requests reading them.
//! That engine's KV events follow: a subscriber of the case hands each batch
//! of its stream to the sink, and its row's index holds every block the
//! events left the engine holding. Then one engine is killed: the first dial
//! that hears it dead writes it, and the pool reads its rows unhealthy and
//! every other row healthy. Next, an engine stops answering with its sockets
//! open: a probe turn waits out its dial bound, writes it and goes on. Last,
//! the gateway sends requests of its own and reports each on its session:
//! the store holds the completion of the engine that answered, none of class
//! ok for the dead and the hung one, and nothing of it for another gateway;
//! a frame that reads wrong ends the session of the gateway that sent it.
//!
//! The gateway asks its first decisions end to end: with a preset of the
//! build applied in the case's process, since no call applies a spec yet,
//! the reference client asks on its decide stream and each pick names a
//! rank of the fleet. The call's bounds are the client's own door: an id at
//! the range's end is answered, a frame past the size leaves cut at its
//! tail, and a frame the call refuses ends the stream, the next ask opening
//! a new one. Then the decisions follow the fleet, each read from the
//! client's answers alone: a load-led pick leaves the engine that holds
//! requests, a rotation names each row in turn, a routing key returns to its
//! worker and a retry leaves it, and no decision names the dead engine.
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
    collections::{HashMap, HashSet},
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
use prost::Message as _;
use smg_policy_controller::{
    client::{self, blocking, Answer, Client},
    controller::{State, PROTOCOL_VERSION},
    proto::{
        completed::Class, decision::Outcome, identity::Form, principal, released, Completed,
        DecideRequest, Decision, Identity, Released, RequestFacts, ShedReason, Side,
    },
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

/// The largest `DecideRequest` as encoded, as the contract states it.
const FRAME: usize = 64 * 1024;

/// An ask for the fleet's model named `id`. It brings the budget of a
/// client's own patience: an ask that states none has none left.
fn ask(id: &[u8]) -> DecideRequest {
    DecideRequest {
        request_id: id.to_vec(),
        facts: Some(RequestFacts {
            model: MODEL.to_owned(),
            budget_micros: 5_000_000,
            ..RequestFacts::default()
        }),
        constraints: None,
    }
}

/// The asks an assertion makes where one draw could hide what it reads.
const ASKS: u64 = 64;

/// A pick's first target: the row of the fleet it names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Picked {
    engine: usize,
    rank: u16,
    /// The row's id, as a retry names it.
    id: u64,
}

/// The `n`th ask of the run named `run`, with an id and a seed of its own.
fn nth(run: &str, n: u64) -> DecideRequest {
    let mut ask = ask(format!("{run}-{n}").as_bytes());
    if let Some(facts) = &mut ask.facts {
        facts.seed = n;
    }
    ask
}

/// `ask` under the routing key `key`.
fn keyed(mut ask: DecideRequest, key: u64) -> DecideRequest {
    if let Some(facts) = &mut ask.facts {
        facts.session_key = Some(Identity {
            form: Some(Form::Hash(key)),
        });
    }
    ask
}

/// Asks, and reads the pick's first target as the row of the fleet it
/// names. A pick that carries a lease is refused: the case releases none.
async fn picked(harness: &mut Harness, ask: DecideRequest) -> Result<Picked, String> {
    let answer = harness.client().decide(ask).await;
    let Ok(Answer::Decision(Decision {
        leases,
        outcome: Some(Outcome::Pick(pick)),
        ..
    })) = &answer
    else {
        return Err(format!("no pick: {answer:?}"));
    };
    // The asserts by count lean on a served decision booking nothing: once
    // a decision carries a lease, a pick releases it here, before the next
    // ask reads the rows.
    if !leases.is_empty() {
        return Err(format!(
            "a served decision carries a lease, released by no ask of this case: {leases:?}"
        ));
    }
    let target = pick.targets.first().ok_or("a pick with no target")?;
    match harness.named(target.id) {
        Some((engine, rank)) if target.dp_rank == u32::from(rank) => Ok(Picked {
            engine,
            rank,
            id: target.id,
        }),
        _ => Err(format!("a target that is no row of the fleet: {target:?}")),
    }
}

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

        // Decide. A preset of the build is applied in the case's process,
        // since no call applies a spec yet. The gateway asks on its decide
        // stream, and the pick names a rank of the fleet.
        harness
            .apply("random")
            .expect("a preset of this build applies");
        picked(&mut harness, ask(b"first"))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert!(
            harness.term_floor() >= leader.term,
            "an answer's term never lowers the floor"
        );
        // The call's bounds are the client's own door. An id at the end of
        // its range is answered.
        picked(&mut harness, ask(&[7; 256]))
            .await
            .expect("a pick answers an ask over a ready fleet");
        // A frame past the size leaves cut at its tail, and the cut is
        // counted: eight bytes a hash.
        let mut long = ask(b"long");
        let facts = long.facts.as_mut().expect("the ask states facts");
        facts.block_hashes = (0u64..).take(8_300).collect();
        let over = long.encoded_len() - FRAME;
        picked(&mut harness, long)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            harness.client().hashes_cut(),
            over.div_ceil(8),
            "{over} bytes over"
        );
        // A frame the call refuses ends the stream by the field's name, and
        // the next ask opens a new one.
        let mut wrong = ask(b"wrong");
        wrong.facts.as_mut().expect("the ask states facts").phase = 99;
        let refused = harness.client().decide(wrong).await;
        assert!(
            matches!(&refused, Err(client::Error::Status(status))
                if status.code() == Code::InvalidArgument
                    && status.message().contains("facts.phase")),
            "{refused:?}"
        );
        picked(&mut harness, ask(b"after"))
            .await
            .expect("a pick answers an ask over a ready fleet");

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

        // A load-led decision follows the loads the engines told: under the
        // least-load preset no pick names the engine that holds requests.
        harness
            .apply("least_load")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("load", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "a load-led pick leaves the engine that holds requests: {pick:?}"
            );
        }

        // Two draws compared by the loads the engines told: under the
        // power-of-two preset no pick names the engine that holds requests.
        // Unread, the loads leave the busy row one pick in six, so 64 asks
        // miss it less than once in 100,000 runs.
        harness
            .apply("power_of_two")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("two", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, busy,
                "of two draws the better load is taken: {pick:?}"
            );
        }

        // The first row the filters keep, in the host's order: under the
        // passthrough preset every pick names the view's first row, and an
        // ask that names it tried is answered with the view's second.
        harness
            .apply("passthrough")
            .expect("a preset of this build applies");
        let pool = harness.view(MODEL, Role::Single).expect("the view");
        let listed: Vec<(usize, u16)> = pool.iter().map(|row| (row.engine, row.rank)).collect();
        let first = picked(&mut harness, nth("through", 0))
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            Some((first.engine, first.rank)),
            listed.first().copied(),
            "a passthrough pick is the first row the host lists: {first:?} of {listed:?}"
        );
        for n in 1..8 {
            let again = picked(&mut harness, nth("through", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, first,
                "a passthrough pick is one row whatever its seed"
            );
        }
        let mut retry = nth("through", 8);
        let facts = retry.facts.as_mut().expect("the ask states facts");
        facts.tried = vec![first.id];
        facts.attempt = 1;
        let next = picked(&mut harness, retry)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_eq!(
            Some((next.engine, next.rank)),
            listed.get(1).copied(),
            "past its tried first row a passthrough pick is the second: {next:?} of {listed:?}"
        );

        // A rotation names every row before it names one again: under the
        // round-robin preset two turns of asks name each row twice.
        harness
            .apply("round_robin")
            .expect("a preset of this build applies");
        let mut turns: HashMap<Picked, u32> = HashMap::new();
        for n in (0u64..).take(2 * ranks) {
            let pick = picked(&mut harness, nth("turn", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            *turns.entry(pick).or_default() += 1;
        }
        assert!(
            turns.len() == ranks && turns.values().all(|asks| *asks == 2),
            "two turns of a rotation name each row twice: {turns:?}"
        );

        // The KV events follow the fleet: the busy engine's held requests
        // stored blocks, its stream's batches reached its rank's cursor, and
        // its row's index holds every block the engine holds.
        let stored = within(
            "the busy engine's blocks reach its row's index",
            PROBE_INTERVAL + ROOM,
            || {
                harness.heard(busy).first().is_some_and(|row| {
                    row.indexed > 0 && row.cursor.is_some_and(|cursor| cursor.applied >= 1)
                })
            },
        )
        .await
        .map_err(|bound| {
            format!(
                "{bound}; what refuses its stream: {:?}",
                harness.refused(busy)
            )
        })
        .expect("the KV events follow");
        let heard = harness.heard(busy);
        let fed = heard.first().expect("the busy engine's one rank");
        assert!(
            fed.last >= 1
                && fed.cursor.is_some_and(|cursor| {
                    cursor.applied >= fed.last && !cursor.replay_pending && !cursor.degraded
                }),
            "the cursor stands at what the stream delivered: {fed:?}"
        );
        assert!(
            fed.applied >= 1 && fed.indexed == fed.held,
            "the index holds each block the engine's events left it holding: {fed:?}"
        );
        assert_eq!(
            fed.counts,
            Some(LaneCounts::default()),
            "no store came without its parent and no block was refused: {fed:?}"
        );
        for engine in engines.iter().filter(|engine| **engine != busy) {
            for idle in harness.heard(*engine) {
                assert!(
                    idle.applied == 0 && idle.indexed == 0 && idle.held == 0,
                    "an engine that stored no block published none: engine {engine} {idle:?}"
                );
            }
        }
        assert_eq!(harness.gaps(), 0, "no stream named a gap");
        assert_eq!(
            harness.snapshot_chunks(),
            0,
            "an engine's own stream carries no snapshot chunk"
        );

        // A keyed ask returns to its worker: under the ring preset one
        // routing key names one row at each ask, and 64 keys do not all
        // share it. On a ring of even shares a correct ring puts 64 keys
        // on one row less than once in 10^40 runs.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let owner = picked(&mut harness, keyed(nth("key", 0), 1))
            .await
            .expect("a pick answers an ask over a ready fleet");
        for n in 1..8 {
            let again = picked(&mut harness, keyed(nth("key", n), 1))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(again, owner, "one routing key returns to one worker");
        }
        let mut owners = HashSet::from([owner]);
        for n in 1..ASKS {
            let other = picked(&mut harness, keyed(nth("keys", n), n + 1))
                .await
                .expect("a pick answers an ask over a ready fleet");
            owners.insert(other);
        }
        assert!(
            owners.len() > 1,
            "{ASKS} routing keys do not share one worker: {owners:?}"
        );
        // A retry that names the worker it tried is answered with another.
        let mut retry = keyed(nth("retry", 0), 1);
        let facts = retry.facts.as_mut().expect("the ask states facts");
        facts.tried = vec![owner.id];
        facts.attempt = 1;
        let next = picked(&mut harness, retry)
            .await
            .expect("a pick answers an ask over a ready fleet");
        assert_ne!(next, owner, "a retry leaves the worker it names as tried");

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
        // No decision names the dead engine: from the probe that heard it
        // dead its rows are no candidates.
        harness
            .apply("random")
            .expect("a preset of this build applies");
        for n in 0..ASKS {
            let pick = picked(&mut harness, nth("dead", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_ne!(
                pick.engine, dead,
                "a dead engine is named by no decision: {pick:?}"
            );
        }

        // The dead engine's event stream broke with it, and its subscriber
        // keeps what refuses it since: the dial of an engine that is gone.
        within(
            "a dead engine's stream is refused at its dial",
            PROBE_INTERVAL + DIAL_TIMEOUT + ROOM,
            || {
                harness
                    .refused(dead)
                    .is_some_and(|refusal| refusal.starts_with("the dial"))
            },
        )
        .await
        .expect("the subscriber keeps what refuses it");

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
        // One session's frames are applied in their order, so the dead and
        // the hung engine's reports were read by now: neither wrote a
        // completion of class ok for this gateway. A completion of another
        // class writes nothing, so these hold the class each frame carries
        // and nothing more.
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
        // The session tells the status it ended with at each ask, and to
        // each report from the first read on.
        let again = other.session_end().await.expect("the end is told again");
        assert_eq!(
            again.code(),
            Code::InvalidArgument,
            "a second ask reads the status the session ended with: {again:?}"
        );
        let told = other
            .released(Released::default())
            .await
            .expect_err("a report into the ended session is refused");
        assert!(
            matches!(&told, client::Error::Status(status) if status.code() == Code::InvalidArgument),
            "a report into the ended session tells the status it ended with: {told:?}"
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

        // The blocking form's report leaves with no call after it: its
        // runtime drives the connection between calls. The client is held,
        // and asked nothing more, until the store has read its completion:
        // its drop would not wait for a report still queued.
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
            "the blocking form's report leaves with no call after it",
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
        writeln!(out, "FLEET_KV_BATCHES {}", fed.applied).expect("stdout");
        writeln!(out, "FLEET_KV_BLOCKS {}", fed.indexed).expect("stdout");
        writeln!(out, "FLEET_DEATH_WRITTEN_MS {}", written.as_millis()).expect("stdout");
        writeln!(out, "FLEET_DEATH_SEEN_MS {}", seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_HANG_SEEN_MS {}", hung_seen.as_millis()).expect("stdout");
        writeln!(out, "FLEET_WALL_MS {}", wall.as_millis()).expect("stdout");
        writeln!(out, "FLEET_CPU_MS {:.0}", (cpu_seconds() - cpu) * 1_000.0).expect("stdout");
        writeln!(out, "FLEET_RSS_KIB {}", vm_rss_kib()).expect("stdout");

        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
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
                    let decided = client.decide(ask(b"blocking"))?;
                    Ok((leader, client.term_floor(), subject, decided))
                });
            let _ = answered.send(asked.map_err(|error| error.to_string()));
        });
        let (leader, floor, subject, decided) = tokio::time::timeout(ROOM, answer)
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
        // Nothing was applied to this controller: no generation binds the
        // model, and the answer is the engine's own shed.
        assert!(
            matches!(&decided, Answer::Decision(Decision { outcome: Some(Outcome::Shed(shed)), .. })
                if shed.reason() == ShedReason::PolicyDeclined && shed.emptied_by == "binding"),
            "{decided:?}"
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
