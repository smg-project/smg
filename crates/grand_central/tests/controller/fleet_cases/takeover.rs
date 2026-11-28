//! A standby takes the fleet when its leader stops.

use std::collections::HashSet;

use smg_grand_central::{
    document::PolicySpec, engine::Role, policies::power_of_two::PowerOfTwoMetric,
};

use crate::{
    fleet::{within, Harness, GATEWAY, POLL_INTERVAL, PROBE_INTERVAL, ROOM},
    given_up, kept, keyed, nth, picked, ASKS, CASE, FLEET, HELD, MODEL,
};

/// A pair's case in this harness: a standby takes the fleet when its leader
/// stops. Two controllers stand over one lease. The fleet registers at the
/// leader, and the standby holds every rank ready and no load report, under
/// the id the leader gave it, by the leader's log alone. The leader stops
/// and the standby leads under a higher term. A gateway that turns to it
/// is answered over the same rows: each routing key is back on the worker
/// it had, a request in flight at the stop is given up at no cost and
/// booked nowhere twice, and the promoted controller's own load monitor
/// reads the engines.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(
    not(target_os = "linux"),
    ignore = "the standby binds 127.0.0.2: off Linux that takes a loopback alias"
)]
async fn a_standby_takes_the_fleet_when_its_leader_stops() {
    let case = async {
        let dir = tempfile::tempdir().expect("a temp dir");
        let (mut harness, standby) = Harness::start_pair(&dir.path().join("lease"))
            .await
            .expect("a leader and its standby start");
        let led = harness.term_floor();

        // The fleet registers at the leader and turns ready there.
        let shapes = &FLEET[..3];
        let engines = harness.register(shapes).await.expect("the fleet registers");
        let busy = engines[1];
        let ranks = shapes.len();
        harness.feed();
        within(
            "every rank is healthy and holds a load report",
            PROBE_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.len() == ranks && pool.iter().all(|row| row.healthy && row.load.is_some())
                })
            },
        )
        .await
        .expect("the fleet turns ready");

        // The standby follows: the leader's log alone brings it every rank,
        // ready, under the id the leader gave it, and no load report.
        within("the standby holds every rank of the fleet", ROOM, || {
            harness.followed(&standby) == Ok(ranks)
        })
        .await
        .expect("the standby follows its leader");

        // A routing key has its worker under the ring preset, and one
        // request is in flight: its lease is kept and dispatched.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        let mut homes = Vec::new();
        for key in 1..=ASKS {
            let home = picked(&mut harness, keyed(nth("home", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            homes.push(home);
        }
        let (_, in_flight) = kept(&mut harness, nth("flight", 0))
            .await
            .expect("a pick answers an ask over a ready fleet");

        // The leader stops, the planned stop. The standby takes the lease
        // and leads under a term of its own, and the gateway that turns to
        // it is welcomed there.
        harness
            .hand_over(standby)
            .await
            .expect("the standby takes the lead");
        assert_eq!(harness.welcome().controller_id, "fleet-standby");
        assert!(
            harness.term_floor() > led,
            "a promoted standby answers under a term above its leader's"
        );
        let pool = harness.pool(MODEL, Role::Single).expect("the pool");
        assert!(
            pool.len() == ranks && pool.iter().all(|row| row.healthy),
            "a promoted standby holds its leader's rows, each ready: {pool:?}"
        );

        // No document crossed with the log: the case applies the one its
        // leader held, as it did there. Each routing key is back on the
        // worker it had, named by the id the leader gave the row.
        harness
            .apply("consistent_hashing")
            .expect("a preset of this build applies");
        for (key, home) in (1..=ASKS).zip(&homes) {
            let again = picked(&mut harness, keyed(nth("again", key), key))
                .await
                .expect("a pick answers an ask over a ready fleet");
            assert_eq!(
                again, *home,
                "a routing key keeps its worker when a standby takes the lead"
            );
        }

        // The request in flight at the stop is booked nowhere twice. Its
        // lease was the stopped leader's: the promoted controller holds no
        // lease of the gateway, takes the release with no error, and by
        // the fewest requests a round of asks, each kept, names every row
        // once.
        assert_eq!(
            harness.leases(GATEWAY),
            0,
            "a promoted standby holds no lease its leader issued"
        );
        given_up(&mut harness, &[in_flight], 0)
            .await
            .expect("a lease of the leader that stopped is given up at no cost");
        let fewest = PolicySpec::power_of_two(PowerOfTwoMetric::LeastRequests)
            .expect("a form of this build's preset");
        harness
            .apply_document(fewest)
            .expect("a form of this build's preset applies");
        let mut round = HashSet::new();
        let mut leases = Vec::new();
        for n in (0u64..).take(ranks) {
            let (row, lease) = kept(&mut harness, nth("round", n))
                .await
                .expect("a pick answers an ask over a ready fleet");
            round.insert(row);
            leases.push(lease);
        }
        assert_eq!(
            round.len(),
            ranks,
            "a lease of the leader that stopped books no row of the promoted one: {round:?}"
        );
        given_up(&mut harness, &leases, 0)
            .await
            .expect("the leases given up leave the file");

        // The promoted controller polls the engines itself from here on:
        // requests held on one engine reach its row, and nobody else writes
        // a load now.
        harness
            .hold(busy, HELD as usize)
            .await
            .expect("the held requests");
        within(
            "the busy engine's row reads its held requests",
            POLL_INTERVAL + POLL_INTERVAL + ROOM,
            || {
                harness.pool(MODEL, Role::Single).is_ok_and(|pool| {
                    pool.iter().any(|row| {
                        row.engine == busy && row.load.is_some_and(|load| load.running == HELD)
                    })
                })
            },
        )
        .await
        .expect("a promoted standby's own load monitor reads the engines");

        harness.stop().await.expect("a clean stop");
    };
    tokio::time::timeout(CASE, Box::pin(case))
        .await
        .expect("the case ends inside its outer bound");
}
