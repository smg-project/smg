//! The end-to-end case, the harness first: a controller on ephemeral ports,
//! a fleet of mock engines and the generated client, in one process. The
//! fleet registers; the controller's own load driver polls it; a probe turn
//! of the case asks each engine its health and writes what it heard; and
//! the pool is read back as a decision's view reads it: every rank a row,
//! healthy by its probe, its load from a poll, an engine that holds
//! requests reading them. Then one engine is killed: the first dial that
//! hears it dead writes it, and the pool reads its rows unhealthy and every
//! other row healthy. Last, an engine stops answering with its sockets
//! open: a probe turn waits out its dial bound, writes it and goes on.
//!
//! No decision is asked yet: the service answers `Leader` alone, and the
//! case asserts that call end to end. The decisions take the views' place
//! when the service answers them.
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
    time::{Duration, Instant},
};

use engines::cpu_seconds;
use fleet::{within, Harness, Shape, DIAL_TIMEOUT, POLL_INTERVAL, PROBE_INTERVAL, ROOM};
use net::vm_rss_kib;
use smg_policy_controller::{
    store::WorkerStatus,
    view::{Role, SignalSource},
};

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

        // Ask. The one call the service answers today, end to end.
        let leader = harness.leader().await.expect("the leader call");
        assert!(leader.is_leader, "a lone controller leads: {leader:?}");
        assert_eq!(leader.controller_id, "fleet-controller");
        assert!(
            leader.term >= 1,
            "a leader answers under a term: {leader:?}"
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
        // the engine's row not ready, and no other row moves.
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
