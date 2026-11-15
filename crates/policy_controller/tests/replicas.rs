//! The controller's replicas end to end (design 5.3): three controllers,
//! each this package's binary in its own process, on ephemeral ports over
//! one lease file, asked who leads by the generated client. One leads and
//! the others name it by its id and its dialable address; the leader is
//! killed, and a follower takes the lease under the next term once the
//! record stood still for its duration; that leader is stopped, releases the
//! lease and the last replica takes it. The case holds the run to order and
//! terms, never to a clock: across every answer, no call sent after a
//! replica answered as the leader under a term is answered as the leader
//! under a lower one, one replica's terms never fall, and the three logs
//! hold one take a term. The promotion's time is read and printed; its bound
//! is the settings' own with room for a loaded host.
//!
//! No stub server: the client is the generated one and the server is the
//! binary. The two cells at the end are the Numbers table's rows, run by
//! name.
#![expect(
    clippy::expect_used,
    reason = "the case's helpers run inside the test; a failure there is the test's failure"
)]

#[expect(
    dead_code,
    reason = "this binary reads the admin pages alone; the other helpers are the listeners' tests'"
)]
#[path = "common/net.rs"]
mod net;
#[expect(
    dead_code,
    reason = "the replicas over a Lease object are the Kubernetes cell's"
)]
#[path = "common/replicas.rs"]
mod replicas;

use std::{
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use net::http_get;
use replicas::{Lease, Record, Replica, ROOM};
use smg_policy_controller::{
    elector::{ElectionSettings, FileLease, LeaseBackend, LeaseRecord},
    proto::{
        policy_controller_client::PolicyControllerClient, GetPolicySpecRequest, SnapshotRequest,
    },
};
use tonic::Code;

/// The order of the design's settings at a fifth of their size: a take
/// after a kill lands in under a second on an idle host.
fn short() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_millis(600),
        renew_period: Duration::from_millis(200),
        renew_deadline: Duration::from_millis(400),
    }
}

/// A stopped replica's listeners stay open a second: it answers `Leader`
/// through its own hand-over.
const SETTLE_SECS: u64 = 1;

fn start(id: &str, lease: &Path, settings: ElectionSettings) -> Replica {
    Replica::start(id, Lease::File(lease), settings, SETTLE_SECS).expect("the replica starts")
}

/// One leader, named by every replica: its place and the term.
async fn agreed(record: &mut Record, replicas: &mut [Replica]) -> (usize, u64) {
    record
        .agreed(replicas)
        .await
        .expect("the replicas agree on a leader")
}

/// The first answer as the leader under a term above `term`.
async fn promoted(
    record: &mut Record,
    replicas: &mut [Replica],
    term: u64,
    within: Duration,
) -> replicas::Answer {
    record
        .promoted(replicas, term, within)
        .await
        .expect("a replica takes the lease")
}

/// The body of a replica's readiness probe.
async fn readiness(replica: &Replica) -> (u16, String) {
    http_get(replica.admin, "/readiness")
        .await
        .expect("the readiness probe answers")
}

/// The record in the lease file now.
fn stored(lease: &Path) -> LeaseRecord {
    FileLease::new(lease)
        .read()
        .expect("the lease file reads")
        .expect("the lease file holds a record")
}

fn millis(duration: Duration) -> u128 {
    duration.as_millis()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn three_replicas_elect_one_leader_and_hand_over_at_its_kill_and_at_its_stop() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let settings = short();
    let within = settings.duration + settings.renew_period + ROOM;
    let mut replicas: Vec<Replica> = ["replica-a", "replica-b", "replica-c"]
        .iter()
        .map(|id| start(id, &lease, settings))
        .collect();
    let mut record = Record::default();
    let mut out = std::io::stdout();

    // One leader. The two others name it by its id and its dialable
    // address, the holder identity of the lease's record, and each probe
    // says its replica's role. A host loaded enough to move the lease
    // between the samples is read again, and so is one that held the leader
    // past its own deadline between them: its probe says follower until its
    // next renewal lands, under the same term.
    let settled = Instant::now() + ROOM;
    let (first, term) = loop {
        let (first, term) = agreed(&mut record, &mut replicas).await;
        let mut probes = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            probes.push(readiness(replica).await);
        }
        let held = stored(&lease);
        if agreed(&mut record, &mut replicas).await != (first, term) {
            continue;
        }
        assert_eq!(held.holder, replicas[first].identity());
        assert_eq!(held.term, term);
        let roles: Vec<(u16, String)> = (0..replicas.len())
            .map(|place| {
                let role = if place == first { "leader" } else { "follower" };
                (200, format!(r#"{{"status":"ready","role":"{role}"}}"#))
            })
            .collect();
        if probes == roles {
            break (first, term);
        }
        assert!(
            Instant::now() < settled,
            "the probes never said the roles the answers named: {probes:?}"
        );
    };

    // The scopes over the wire: a listener denies the calls of the scopes
    // it does not mount. Those of its own it refuses as unauthenticated: no
    // listener of these replicas takes an identity as told.
    let follower = &replicas[(first + 1) % replicas.len()];
    for (listener, read, snapshot) in [
        (
            follower.gateway,
            Code::PermissionDenied,
            Code::PermissionDenied,
        ),
        (
            follower.control,
            Code::Unauthenticated,
            Code::Unauthenticated,
        ),
        (
            follower.replication,
            Code::PermissionDenied,
            Code::Unauthenticated,
        ),
    ] {
        let mut client = PolicyControllerClient::connect(format!("http://{listener}"))
            .await
            .expect("the listener accepts");
        let refused = client
            .get_policy_spec(GetPolicySpecRequest::default())
            .await
            .expect_err("no step serves the call");
        assert_eq!(refused.code(), read, "GetPolicySpec on {listener}");
        let refused = client
            .get_snapshot(SnapshotRequest::default())
            .await
            .expect_err("no step serves the call");
        assert_eq!(refused.code(), snapshot, "GetSnapshot on {listener}");
    }

    // THE KILL. The lease is not released: a follower takes it once the
    // record stood still for its duration by that follower's own clock.
    let mut killed = replicas.remove(first);
    killed.kill().expect("SIGKILL");
    let at = Instant::now();
    let successor = promoted(&mut record, &mut replicas, term, within).await;
    let after_kill = successor.returned.duration_since(at);
    writeln!(out, "PROMOTION_AFTER_KILL_MS {}", millis(after_kill)).expect("stdout");
    let (mut leader, mut term) = agreed(&mut record, &mut replicas).await;
    assert!(term > successor.info.term || replicas[leader].id == successor.replica);
    assert_eq!(readiness(&replicas[leader]).await.0, 200);

    // THE STOP. The leader closes its fence, releases the lease and exits
    // clean; the last replica takes the released lease. A leader stalled
    // past the duration before its signal has lost the lease and none to
    // release: its log says the lease moved, that stop tests nothing, and
    // the leader after it is stopped beside a fresh follower.
    let mut gone = vec![killed];
    loop {
        let mut stopped = replicas.remove(leader);
        stopped.terminate().expect("SIGTERM");
        let at = Instant::now();
        let successor = promoted(&mut record, &mut replicas, term, within).await;
        let after_stop = successor.returned.duration_since(at);
        let status = stopped.exited().expect("the stopped leader exits");
        assert!(
            status.success(),
            "a stopped leader exits clean, read {status}:\n{}",
            stopped.log()
        );
        let released = stopped.logged("the lease is released");
        assert!(
            released || stopped.lost_its_lease(),
            "the stopped leader released its lease:\n{}",
            stopped.log()
        );
        if released {
            writeln!(out, "PROMOTION_AFTER_STOP_MS {}", millis(after_stop)).expect("stdout");
        }
        gone.push(stopped);
        if released {
            break;
        }
        assert!(
            gone.len() < 4,
            "three leaders in a row lost the lease before their stop"
        );
        replicas.push(start(
            &format!("replica-{}", gone.len() + 2),
            &lease,
            settings,
        ));
        (leader, term) = agreed(&mut record, &mut replicas).await;
    }
    let (last, term) = agreed(&mut record, &mut replicas).await;
    assert_eq!(last, 0, "one replica is left, and it leads");
    assert_eq!(
        readiness(&replicas[last]).await,
        (200, r#"{"status":"ready","role":"leader"}"#.to_owned())
    );
    assert_eq!(stored(&lease).holder, replicas[last].identity());
    assert_eq!(stored(&lease).term, term);

    // The election's promise over every answer of the run, and one take a
    // term across every replica's log: 1, 2, 3 and on, each once.
    record.check().expect("the election's promise holds");
    // The last leader lives on: its take's line reaches the log a moment
    // after its answer turned.
    assert!(
        replicas[last]
            .logs(|leader| leader.takes().last() == Some(&term))
            .await,
        "the last leader logged no take of term {term}:\n{}",
        replicas[last].log()
    );
    let mut takes: Vec<u64> = gone
        .iter()
        .chain([&replicas[last]])
        .flat_map(Replica::takes)
        .collect();
    takes.sort_unstable();
    let terms: Vec<u64> = (1..=term).collect();
    assert_eq!(takes, terms, "one take a term, by every replica's log");
    assert!(
        term >= 3,
        "three takes at least: the first, the kill's, the stop's"
    );
    writeln!(out, "LEADER_ANSWERS_RECORDED {}", record.len()).expect("stdout");
}

/// The middle of a sorted list and its last entry.
fn median_and_worst(samples: &mut [u128]) -> (u128, u128) {
    samples.sort_unstable();
    (
        samples[samples.len() / 2],
        samples.last().copied().unwrap_or_default(),
    )
}

/// A delay under `period` that moves with the round, so the kills, the
/// stops and the starts fall across the leader's renew period and not in
/// step with it: steps of the golden ratio's fraction cover a period
/// evenly.
fn spread(round: u64, offset: u64, period: Duration) -> Duration {
    let thousandths = (round * 618 + offset) % 1000;
    period * u32::try_from(thousandths).unwrap_or_default() / 1000
}

/// A day in microseconds: a wall stamp's time of day, as a log line's.
const DAY_MICROS: u64 = 86_400_000_000;

fn rounds(variable: &str, default: u64) -> u64 {
    std::env::var(variable)
        .ok()
        .and_then(|rounds| rounds.parse().ok())
        .unwrap_or(default)
}

/// The Numbers table's failover rows at the design's default settings
/// (3 s, 1 s, 2 s) or at the timings the environment names: the
/// promotion's time over `SMG_REPLICAS_ROUNDS` kills
/// and as many stops (default 5), three replicas kept by a fresh one after
/// each. The replicas start out of step, and each kill, stop and fresh
/// start falls at another point of the renew period. A kill is read three
/// ways: from the kill to the first answer as the leader, from the dead
/// leader's last renewal to the take (the two wall stamps of the lease's
/// records), and from the dead leader's fence closing (its last renewal
/// plus the renew deadline) to the take. A stop is read three ways: from
/// the signal to the successor's first answer as the leader, by this
/// test's clock; from the signal to the release, by the stopped leader's
/// own log; and from that signal to the take, by the stamp the successor
/// wrote in the lease, which no wait of this test's calls can stretch. A
/// measurement, run by name with `--ignored`; it asserts the election's
/// promise alone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn promotion_times_over_repeated_kills_and_stops() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let within = settings.duration + settings.renew_period + ROOM;
    let rounds = rounds("SMG_REPLICAS_ROUNDS", 5);
    let mut next = 0u64;
    let mut fresh = |lease: &Path| {
        next += 1;
        start(&format!("replica-{next}"), lease, settings)
    };
    let period = settings.renew_period;
    let mut replicas = Vec::new();
    for _ in 0..3 {
        replicas.push(fresh(&lease));
        tokio::time::sleep(period / 3).await;
    }
    let mut record = Record::default();

    let (mut from_kill, mut from_renewal, mut from_close) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..rounds {
        let (leader, term) = agreed(&mut record, &mut replicas).await;
        tokio::time::sleep(spread(round, 0, period)).await;
        let mut killed = replicas.remove(leader);
        killed.kill().expect("SIGKILL");
        let at = Instant::now();
        // The dead leader's last renewal stands in the file until the take.
        let last = stored(&lease);
        let successor = promoted(&mut record, &mut replicas, term, within).await;
        let taken = stored(&lease);
        let kill_ms = millis(successor.returned.duration_since(at));
        let renewal_ms = u128::from(
            taken
                .acquired_at_micros
                .saturating_sub(last.renewed_at_micros),
        ) / 1000;
        let close_ms = renewal_ms.saturating_sub(millis(settings.renew_deadline));
        writeln!(
            out,
            "KILL round={round} term={}->{} from_kill_ms={kill_ms} from_last_renewal_ms={renewal_ms} from_fence_close_ms={close_ms}",
            last.term, taken.term
        )
        .expect("stdout");
        if last.holder == killed.identity() && taken.term == last.term + 1 {
            from_kill.push(kill_ms);
            from_renewal.push(renewal_ms);
            from_close.push(close_ms);
        }
        tokio::time::sleep(spread(round, 500, period)).await;
        replicas.push(fresh(&lease));
    }

    let (mut from_stop, mut to_release, mut to_take) = (Vec::new(), Vec::new(), Vec::new());
    for round in 0..rounds {
        let (leader, term) = agreed(&mut record, &mut replicas).await;
        tokio::time::sleep(spread(round, 0, period)).await;
        let mut stopped = replicas.remove(leader);
        stopped.terminate().expect("SIGTERM");
        let at = Instant::now();
        let successor = promoted(&mut record, &mut replicas, term, within).await;
        let stop_ms = millis(successor.returned.duration_since(at));
        let taken = stored(&lease);
        assert!(stopped
            .exited()
            .expect("the stopped leader exits")
            .success());
        let signal = stopped
            .logged_at_micros("SIGTERM received")
            .expect("the stopped leader logged its signal");
        let Some(released) = stopped.logged_at_micros("the lease is released") else {
            // A leader stalled past the duration before its signal lost
            // the lease and has none to release: no stop's round.
            assert!(
                stopped.lost_its_lease(),
                "the stopped leader logged its release:\n{}",
                stopped.log()
            );
            writeln!(
                out,
                "STOP round={round} void: the leader lost the lease first"
            )
            .expect("stdout");
            tokio::time::sleep(spread(round, 500, period)).await;
            replicas.push(fresh(&lease));
            continue;
        };
        let release_us = u128::from(released.saturating_sub(signal));
        // The take's wall stamp on the clock of the signal's log line.
        let take_ms =
            u128::from((taken.acquired_at_micros % DAY_MICROS).saturating_sub(signal)) / 1000;
        writeln!(
            out,
            "STOP round={round} term={term}->{} from_stop_ms={stop_ms} release_us={release_us} signal_to_take_ms={take_ms}",
            taken.term
        )
        .expect("stdout");
        from_stop.push(stop_ms);
        to_release.push(release_us);
        if taken.term == term + 1 {
            to_take.push(take_ms);
        }
        tokio::time::sleep(spread(round, 500, period)).await;
        replicas.push(fresh(&lease));
    }

    for (name, samples) in [
        ("KILL_TO_LEADER_ANSWER_MS", &mut from_kill),
        ("LAST_RENEWAL_TO_TAKE_MS", &mut from_renewal),
        ("FENCE_CLOSE_TO_TAKE_MS", &mut from_close),
        ("STOP_TO_LEADER_ANSWER_MS", &mut from_stop),
        ("STOP_TO_RELEASE_US", &mut to_release),
        ("STOP_SIGNAL_TO_TAKE_MS", &mut to_take),
    ] {
        let n = samples.len();
        let (median, worst) = median_and_worst(samples);
        let best = samples.first().copied().unwrap_or_default();
        writeln!(
            out,
            "{name} n={n} best={best} median={median} worst={worst}"
        )
        .expect("stdout");
    }
    record.check().expect("the election's promise holds");
    writeln!(out, "LEADER_ANSWERS_RECORDED {}", record.len()).expect("stdout");
}

/// The Numbers table's idle rows at the same timings: the CPU
/// time and the resident size of a leader and of a follower over
/// `SMG_REPLICAS_IDLE_SECS` seconds (default 60) with no client. A
/// measurement, run by name with `--ignored`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn cpu_and_resident_memory_of_an_idle_leader_and_an_idle_follower() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let secs = rounds("SMG_REPLICAS_IDLE_SECS", 60);
    let mut replicas = vec![
        start("replica-1", &lease, settings),
        start("replica-2", &lease, settings),
    ];
    let mut record = Record::default();
    let (leader, term) = agreed(&mut record, &mut replicas).await;
    // The starts' own work is out of the window.
    tokio::time::sleep(Duration::from_secs(2)).await;
    let before: Vec<u64> = replicas.iter().map(Replica::cpu_nanos).collect();
    tokio::time::sleep(Duration::from_secs(secs)).await;
    for (place, replica) in replicas.iter().enumerate() {
        let role = if place == leader {
            "LEADER"
        } else {
            "FOLLOWER"
        };
        let ran = replica.cpu_nanos().saturating_sub(before[place]);
        writeln!(
            out,
            "IDLE_{role} secs={secs} cpu_us={} cpu_us_per_wall_s={} rss_kib={}",
            ran / 1000,
            ran / 1000 / secs.max(1),
            replica.rss_kib()
        )
        .expect("stdout");
    }
    assert_eq!(
        agreed(&mut record, &mut replicas).await,
        (leader, term),
        "the idle leader kept its lease"
    );
    record.check().expect("the election's promise holds");
}
