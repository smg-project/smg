//! The controller's replicas end to end (design 5.3): three controllers,
//! each this package's binary in its own process, on ephemeral ports over
//! one lease file, asked who leads by the generated client. One leads and
//! the others name it by its id and its dialable address; the leader is
//! killed, and a follower takes the lease under the next term once the
//! record stood still for its duration; that leader is stopped beside the
//! last follower, which leads under the next term; the replica left is
//! stopped alone, releases the lease, and a replica started after it takes
//! the released lease under the next term. The case holds the run to order
//! and terms, never to a clock: across every answer, no call sent after a
//! replica answered as the leader under a term is answered as the leader
//! under a lower one, one replica's terms never fall, and the replicas' logs
//! hold one take a term. The promotion's time is read and printed; its bound
//! is the settings' own with room for a loaded host.
//!
//! A second case kills a lone leader and starts it again under its
//! identity: the new run leads under the term above, after the wait of
//! every candidate.
//!
//! No stub server: the client is the generated one and the server is the
//! binary. The cells at the end are the Numbers table's rows, run by name.
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
    collections::BTreeMap,
    io::Write,
    path::Path,
    time::{Duration, Instant},
};

use net::http_get;
use replicas::{Lease, Record, Replica, ROOM};
use smg_policy_controller::service::{
    elector::{ConditionalWriteOutcome, ElectionSettings, FileLease, LeaseBackend, LeaseRecord},
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

/// What a stopped replica logs when its election thread did not end within
/// the grace period: the release was given up, and the lease expires.
const GIVEN_UP: &str =
    "the election thread outlived the grace period; a lease it holds expires by its duration";

fn start(id: &str, lease: &Path, settings: ElectionSettings) -> Replica {
    Replica::start(id, Lease::File(lease), settings, SETTLE_SECS).expect("the replica starts")
}

/// The address a replica of the restart cases advertises: the holder
/// identity is the id at this address whatever its listeners bind, so a
/// second process started the same way is the first one's restart to the
/// election. Nothing dials it: the cases ask each process where it listens.
const ADVERTISED: &str = "127.0.0.1:9";

fn start_at(id: &str, lease: &Path, settings: ElectionSettings) -> Replica {
    Replica::start_at(id, ADVERTISED, Lease::File(lease), settings, SETTLE_SECS)
        .expect("the replica starts")
}

/// Samples until a replica answers as the leader: that answer.
async fn led(record: &mut Record, replicas: &mut [Replica]) -> replicas::Answer {
    let started = Instant::now();
    loop {
        let answers = record.sample(replicas).await;
        if let Some(answer) = answers.into_iter().find(|answer| answer.info.is_leader) {
            return answer;
        }
        assert!(started.elapsed() < ROOM, "no replica led in {ROOM:?}");
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
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

/// A leader killed and started again under its identity, as a lone
/// controller over its lease file or a container restarted in its pod. The
/// new run waits its earlier run's record out for the record's duration, as
/// every candidate does, and takes the term above: it never leads under the
/// term its earlier run led under. A term is taken once, so what a reader
/// holds under a term came from one run of one store.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_leader_restarted_under_its_identity_leads_under_the_term_above() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let settings = short();
    let mut record = Record::default();
    let mut replicas = vec![start_at("replica-a", &lease, settings)];
    let first = led(&mut record, &mut replicas).await;
    let identity = replicas[0].identity();
    let mut earlier = replicas.remove(0);
    earlier.kill().expect("SIGKILL");
    let killed = Instant::now();
    let last = stored(&lease);
    assert_eq!(
        (last.holder.as_str(), last.term),
        (identity.as_str(), first.info.term)
    );

    // The same id at the same advertised address: the earlier run's record
    // is under this run's own name. The new process answers into a record
    // of its own: it answers term zero until its first look at the lease,
    // which one record over both runs would read as the first run's term
    // falling.
    replicas.push(start_at("replica-a", &lease, settings));
    let mut restarted = Record::default();
    let again = led(&mut restarted, &mut replicas).await;
    assert_eq!(
        again.info.term,
        first.info.term + 1,
        "the restart leads under the term above its earlier run's"
    );
    let waited = again.returned.duration_since(killed);
    assert!(
        waited >= settings.duration,
        "the restart led {waited:?} after the kill, inside the record's duration"
    );
    let taken = stored(&lease);
    assert_eq!(
        (taken.holder.as_str(), taken.term),
        (identity.as_str(), again.info.term)
    );
    assert!(
        taken.acquired_at_micros > last.renewed_at_micros,
        "a take in every field"
    );
    record.check().expect("the election's promise holds");
    restarted
        .check()
        .expect("the election's promise holds after the restart");
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
            // Read again, under the loop's own clock: a lease that moves
            // between every two looks is this case's failure, not its wait.
            assert!(
                Instant::now() < settled,
                "the lease kept moving between two looks at it"
            );
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
            Code::PermissionDenied,
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
            .expect_err("no step or no follower serves the call");
        assert_eq!(refused.code(), snapshot, "GetSnapshot on {listener}");
    }

    // The first leader took a lease no record stood for: its fence opened
    // between two steps, and its line of leading comes at the step after.
    // The kill waits for that line, so the logs hold one take a term.
    assert!(
        replicas[first]
            .logs(|leader| leader.takes().last() == Some(&term))
            .await,
        "the first leader logged no take of term {term}:\n{}",
        replicas[first].log()
    );

    // THE KILL. The lease is not released: a follower takes it once the
    // record stood still for its duration by that follower's own clock.
    let mut killed = replicas.remove(first);
    killed.kill().expect("SIGKILL");
    let at = Instant::now();
    let successor = promoted(&mut record, &mut replicas, term, within).await;
    let after_kill = successor.returned.duration_since(at);
    writeln!(out, "PROMOTION_AFTER_KILL_MS {}", millis(after_kill)).expect("stdout");
    let (leader, term) = agreed(&mut record, &mut replicas).await;
    assert!(term > successor.info.term || replicas[leader].id == successor.replica);
    assert_eq!(readiness(&replicas[leader]).await.0, 200);

    // THE HAND-OVER AT A STOP. The leader is stopped beside the live
    // follower, and the follower leads under a term above: it takes the
    // released lease, or the one a leader held back past the duration lost
    // before it read its stop. That leader has none to release and logs
    // nothing, so its log cannot tell the two apart, and this stop is read
    // by order alone: who leads next.
    let mut gone = vec![killed];
    let mut stopped = replicas.remove(leader);
    stopped.terminate().expect("SIGTERM");
    promoted(&mut record, &mut replicas, term, within).await;
    let status = stopped.exited().expect("the stopped leader exits");
    assert!(
        status.success(),
        "a stopped leader exits clean, read {status}:\n{}",
        stopped.log()
    );
    gone.push(stopped);

    // THE RELEASE. The replica that is left runs alone, so nothing but its
    // own release can move the lease. It closes its fence, releases the
    // lease and exits clean, and the stop is read from what it left: the
    // release in its log, and the record released under its term. A release
    // given up at the grace period is in the log too: that stop tests
    // nothing, and the replica started after it is stopped in its turn.
    let term = loop {
        let (alone, term) = agreed(&mut record, &mut replicas).await;
        let mut stopped = replicas.remove(alone);
        stopped.terminate().expect("SIGTERM");
        let status = stopped.exited().expect("the stopped leader exits");
        assert!(
            status.success(),
            "a stopped leader exits clean, read {status}:\n{}",
            stopped.log()
        );
        let released = stopped.logged("the lease is released");
        assert!(
            released || stopped.logged(GIVEN_UP),
            "the stopped leader released its lease:\n{}",
            stopped.log()
        );
        let left = stored(&lease);
        assert!(
            !released || (left.is_released() && left.term == term),
            "a release leaves the record released under term {term}, read {left:?}"
        );
        gone.push(stopped);
        assert!(
            released || gone.len() < 5,
            "three leaders in a row gave their release up at the grace period"
        );
        replicas.push(start(
            &format!("replica-{}", gone.len() + 1),
            &lease,
            settings,
        ));
        if released {
            break term;
        }
    };

    // The replica started after the release takes the released lease under
    // the next term and leads alone. A host that held it past its own
    // deadline between the answer and the probe reads follower until its
    // next renewal lands, under the same term: the pair is read again.
    let settled = Instant::now() + ROOM;
    let last = loop {
        let (last, taken) = agreed(&mut record, &mut replicas).await;
        assert_eq!(
            taken,
            term + 1,
            "the released lease is taken under the next term"
        );
        let probe = readiness(&replicas[last]).await;
        if probe == (200, r#"{"status":"ready","role":"leader"}"#.to_owned()) {
            break last;
        }
        assert!(
            Instant::now() < settled,
            "the last replica's probe never said leader: {probe:?}"
        );
    };
    let term = term + 1;
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

/// A gateway's term floor above the leader's term (design 5.3), over two
/// replicas and one lease file. The gateway holds a term the lease does not:
/// its `Hello` under that floor is refused, the leader gives the lease up,
/// and the take after it is above the floor, where the gateway's next
/// `Hello` under the same floor is welcomed. A term never goes back while a
/// gateway that saw it lives.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_gateways_floor_above_the_leaders_term_moves_the_lease_above_it() {
    const GATEWAY: &str = "gateway-a";
    const FLOOR: u64 = 7;
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let settings = short();
    let mut record = Record::default();
    let mut replicas = vec![
        start("replica-a", &lease, settings),
        start("replica-b", &lease, settings),
    ];
    let (leader, term) = agreed(&mut record, &mut replicas).await;
    assert!(term < FLOOR, "the first terms are under the floor: {term}");

    assert_eq!(
        replicas[leader].hello_under(GATEWAY, FLOOR).await,
        Err("FailedPrecondition".to_owned()),
        "a floor above the leader's term is refused"
    );

    // The leader gave the lease up under the floor it was shown, and the
    // take after it is above that floor: whichever replica leads there
    // welcomes the gateway under its floor.
    let said = Instant::now();
    let welcomed = loop {
        let mut answers = Vec::with_capacity(replicas.len());
        for replica in &replicas {
            answers.push(replica.hello_under(GATEWAY, FLOOR).await);
        }
        if let Some(term) = answers.iter().find_map(|answer| answer.as_ref().ok()) {
            break *term;
        }
        assert!(
            said.elapsed() < ROOM,
            "no Welcome under the floor in {ROOM:?}: {answers:?}"
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        welcomed > FLOOR,
        "welcomed under term {welcomed}, not above the floor {FLOOR}"
    );
    let (_, after) = agreed(&mut record, &mut replicas).await;
    assert!(after >= welcomed, "{after} after {welcomed}");
    record.check().expect("the election's promise holds");
}

/// The middle of a sorted list and its last entry. A group with no sample
/// (no round ran, or none passed its filter) has neither and is refused by
/// its name.
fn median_and_worst(name: &str, samples: &mut [u128]) -> (u128, u128) {
    assert!(
        !samples.is_empty(),
        "{name}: no sample; a summary needs one round at the least"
    );
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

/// One line a group of a measurement cell: its count, best, median and
/// worst.
fn summaries<const N: usize>(out: &mut impl Write, groups: [(&str, &mut Vec<u128>); N]) {
    for (name, samples) in groups {
        let n = samples.len();
        let (median, worst) = median_and_worst(name, samples);
        let best = samples.first().copied().unwrap_or_default();
        writeln!(
            out,
            "{name} n={n} best={best} median={median} worst={worst}"
        )
        .expect("stdout");
    }
}

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

    summaries(
        &mut out,
        [
            ("KILL_TO_LEADER_ANSWER_MS", &mut from_kill),
            ("LAST_RENEWAL_TO_TAKE_MS", &mut from_renewal),
            ("FENCE_CLOSE_TO_TAKE_MS", &mut from_close),
            ("STOP_TO_LEADER_ANSWER_MS", &mut from_stop),
            ("STOP_TO_RELEASE_US", &mut to_release),
            ("STOP_SIGNAL_TO_TAKE_MS", &mut to_take),
        ],
    );
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

/// The longest run of samples in a row that met a condition, from the
/// run's first sample to its last.
#[derive(Default)]
struct Stretch {
    since: Option<Instant>,
    longest: Duration,
}

impl Stretch {
    fn read(&mut self, met: bool, at: Instant) {
        if met {
            let since = *self.since.get_or_insert(at);
            self.longest = self.longest.max(at.duration_since(since));
        } else {
            self.since = None;
        }
    }
}

/// What a removed lease file costs, at the design's default settings or at
/// the timings the environment names. Over `SMG_REPLICAS_ROUNDS` rounds
/// (default 5) a replica starts over a new lease file and its first answer
/// as the leader is timed from its start; a standby joins; the lease file
/// is removed under the leader, at another point of the renew period each
/// round; and both replicas are asked `Leader` every few milliseconds until
/// one answers as the leader above the lost term and a renew deadline more.
/// A removal is read four ways: the longest run of samples in which both
/// answered as the leader, the longest in which neither did, the removal to
/// the first answer above the lost term, and whether the election's promise
/// held over the round's answers. A measurement, run by name with
/// `--ignored`; it asserts nothing of the election, so it reads a binary
/// that takes a removed record at once as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_lease_file_removed_under_a_live_leader_with_a_standby_looking() {
    lost_under_a_live_leader(Loss::Removed).await;
}

/// The same rounds with the record's holder cleared in place, the term, the
/// stamps and the duration kept, as a hand's patch of the holder leaves it:
/// a write through the store, so no renewal in flight puts the holder back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_holder_cleared_in_place_under_a_live_leader_with_a_standby_looking() {
    lost_under_a_live_leader(Loss::Cleared).await;
}

/// The same rounds with no standby looking: the lease file is removed under
/// a lone live leader and a second replica starts right then. It never saw
/// the lease held, so it takes at once under its own run's first term, and
/// the round reads its own first answer as the leader.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_replica_started_while_a_live_leaders_lease_file_is_gone() {
    lost_under_a_live_leader(Loss::Started).await;
}

/// How a round of the three cells above loses the lease's record.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Loss {
    Removed,
    Cleared,
    Started,
}

impl Loss {
    /// The name on the printed lines.
    fn label(self) -> &'static str {
        match self {
            Loss::Removed => "REMOVED",
            Loss::Cleared => "CLEARED",
            Loss::Started => "STARTED",
        }
    }
}

/// The compare-and-swaps a clearing may lose to renewals before the cell
/// ends by name.
const CLEARINGS: u32 = 100;

/// The rounds of the three cells above.
async fn lost_under_a_live_leader(loss: Loss) {
    let label = loss.label();
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let rounds = rounds("SMG_REPLICAS_ROUNDS", 5);
    let period = settings.renew_period;
    let within = settings.duration + 2 * period + ROOM;
    let (mut first, mut two_ms, mut nobody_ms, mut take_ms) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new());
    let (mut with_two, mut broken) = (0u64, 0u64);
    for round in 0..rounds {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let mut record = Record::default();
        let started = Instant::now();
        let mut replicas = vec![start(&format!("replica-{round}-a"), &lease, settings)];
        agreed(&mut record, &mut replicas).await;
        let first_ms = millis(started.elapsed());
        tokio::time::sleep(spread(round, 333, period)).await;
        let second = format!("replica-{round}-b");
        if loss != Loss::Started {
            replicas.push(start(&second, &lease, settings));
            agreed(&mut record, &mut replicas).await;
            tokio::time::sleep(spread(round, 0, period)).await;
        }
        // The loss is read against the term of the record it moved: a
        // change of leader after the last agreement would leave an agreed
        // term one behind.
        let term = match loss {
            Loss::Cleared => {
                // A compare and swap through the store: a renewal that
                // lands first moves the record, and the clearing is tried
                // again, a bounded number of times.
                let mut store = FileLease::new(&lease);
                (0..CLEARINGS)
                    .find_map(|_| {
                        let held = store.read().expect("the lease file reads");
                        let held = held.expect("the lease file holds a record");
                        let cleared = LeaseRecord {
                            holder: String::new(),
                            ..held.clone()
                        };
                        let written = store.write(Some(&held), &cleared).expect("the clearing");
                        (written == ConditionalWriteOutcome::Done).then_some(held.term)
                    })
                    .expect("a clearing lands within its tries")
            }
            Loss::Removed | Loss::Started => {
                // The removal under the store's own lock, as a writer of
                // the store moves the file: no renewal in flight puts the
                // record back.
                let lock = std::fs::File::create(dir.path().join("lease.lock")).expect("the lock");
                lock.lock().expect("the store's lock");
                let term = stored(&lease).term;
                std::fs::remove_file(&lease).expect("the removal");
                term
            }
        };
        let removed = Instant::now();
        if loss == Loss::Started {
            replicas.push(start(&second, &lease, settings));
        }
        let (mut samples, mut both) = (0u64, 0u64);
        let (mut two, mut nobody) = (Stretch::default(), Stretch::default());
        let mut taken: Option<(Instant, u64)> = None;
        loop {
            let at = Instant::now();
            let answers = record.sample(&mut replicas).await;
            let leaders = answers.iter().filter(|a| a.info.is_leader).count();
            samples += 1;
            both += u64::from(leaders == 2);
            two.read(leaders == 2, at);
            nobody.read(leaders == 0, at);
            if taken.is_none() {
                // A standby takes above the lost term. A replica started
                // after the loss counts from its own run's first term, so
                // its take is read by its name.
                taken = answers
                    .iter()
                    .find(|a| {
                        a.info.is_leader
                            && match loss {
                                Loss::Started => a.replica == second,
                                Loss::Removed | Loss::Cleared => a.info.term > term,
                            }
                    })
                    .map(|a| (a.returned, a.info.term));
            }
            match taken {
                Some((since, _)) if since.elapsed() >= settings.renew_deadline => break,
                None if removed.elapsed() > within => break,
                _ => tokio::time::sleep(Duration::from_millis(2)).await,
            }
        }
        let Some((at, next)) = taken else {
            writeln!(out, "{label} round={round} void: no take in {within:?}").expect("stdout");
            continue;
        };
        let promise = if record.check().is_ok() {
            "held"
        } else {
            broken += 1;
            "broken"
        };
        with_two += u64::from(both > 0);
        let (both_ms, none_ms, to_take_ms) = (
            millis(two.longest),
            millis(nobody.longest),
            millis(at.duration_since(removed)),
        );
        writeln!(
            out,
            "{label} round={round} term={term}->{next} first_start_ms={first_ms} samples={samples} both={both} both_ms={both_ms} nobody_ms={none_ms} to_take_ms={to_take_ms} promise={promise}"
        )
        .expect("stdout");
        first.push(first_ms);
        two_ms.push(both_ms);
        nobody_ms.push(none_ms);
        take_ms.push(to_take_ms);
    }
    let counted = take_ms.len();
    let names =
        ["TWO_LEADERS_MS", "NOBODY_LEADS_MS", "TO_TAKE_MS"].map(|name| format!("{label}_{name}"));
    summaries(
        &mut out,
        [
            ("FIRST_START_TO_LEADER_ANSWER_MS", &mut first),
            (names[0].as_str(), &mut two_ms),
            (names[1].as_str(), &mut nobody_ms),
            (names[2].as_str(), &mut take_ms),
        ],
    );
    writeln!(
        out,
        "{label}_ROUNDS n={counted} with_two_leaders={with_two} promise_broken={broken}"
    )
    .expect("stdout");
}

/// What a leader's restart costs, at the design's default settings or at
/// the timings the environment names. Over `SMG_REPLICAS_ROUNDS` rounds
/// (default 5) a lone leader under an advertised identity is killed, at
/// another point of its renew period each round, and started again at once;
/// the kill to the new run's first answer as the leader is read by this
/// test's clock, with the term before and after. A measurement, run by name
/// with `--ignored`; it asserts nothing of the election, so it reads a
/// binary that leads again under its earlier run's term, or one that takes
/// with no wait, as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn restart_times_of_a_leader_under_its_identity() {
    let dir = tempfile::tempdir().expect("a temp dir");
    let lease = dir.path().join("lease");
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let rounds = rounds("SMG_REPLICAS_ROUNDS", 5);
    let period = settings.renew_period;
    let mut record = Record::default();
    let mut replicas = vec![start_at("replica-a", &lease, settings)];
    let (mut from_kill, mut same_term) = (Vec::new(), 0u64);
    for round in 0..rounds {
        let term = led(&mut record, &mut replicas).await.info.term;
        tokio::time::sleep(spread(round, 0, period)).await;
        let mut earlier = replicas.remove(0);
        earlier.kill().expect("SIGKILL");
        let at = Instant::now();
        replicas.push(start_at("replica-a", &lease, settings));
        let again = led(&mut record, &mut replicas).await;
        let kill_ms = millis(again.returned.duration_since(at));
        same_term += u64::from(again.info.term == term);
        writeln!(
            out,
            "RESTART round={round} term={term}->{} from_kill_ms={kill_ms}",
            again.info.term
        )
        .expect("stdout");
        from_kill.push(kill_ms);
    }
    summaries(
        &mut out,
        [("RESTART_KILL_TO_LEADER_ANSWER_MS", &mut from_kill)],
    );
    writeln!(
        out,
        "RESTART_ROUNDS n={rounds} under_the_earlier_term={same_term}"
    )
    .expect("stdout");
}

/// What the restart's wait is for, at the same timings. Over the rounds a
/// second process is started under a live leader's identity, at another
/// point of the leader's renew period each round, and both are asked
/// `Leader` every few milliseconds for a lease duration and a renew period
/// more. A round is read two ways: the longest run of samples in which both
/// answered as the leader, and whether the second process led at all. A
/// measurement, run by name with `--ignored`; it asserts nothing of the
/// election, so it reads a binary that takes with no wait as well.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_second_process_started_under_a_live_leaders_identity() {
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let rounds = rounds("SMG_REPLICAS_ROUNDS", 5);
    let period = settings.renew_period;
    let (mut two_ms, mut second_led) = (Vec::new(), 0u64);
    for round in 0..rounds {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        let mut record = Record::default();
        let mut replicas = vec![start_at("replica-a", &lease, settings)];
        led(&mut record, &mut replicas).await;
        tokio::time::sleep(spread(round, 0, period)).await;
        replicas.push(start_at("replica-a", &lease, settings));
        let started = Instant::now();
        let (mut samples, mut both, mut led_too) = (0u64, 0u64, false);
        let mut two = Stretch::default();
        while started.elapsed() < settings.duration + period {
            let at = Instant::now();
            let mut leaders = 0;
            for (place, replica) in replicas.iter_mut().enumerate() {
                let leads = replica.leader().await.is_some_and(|info| info.is_leader);
                leaders += usize::from(leads);
                led_too |= leads && place == 1;
            }
            samples += 1;
            both += u64::from(leaders == 2);
            two.read(leaders == 2, at);
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
        second_led += u64::from(led_too);
        writeln!(
            out,
            "TWIN round={round} samples={samples} both={both} both_ms={} second_led={led_too}",
            millis(two.longest)
        )
        .expect("stdout");
        two_ms.push(millis(two.longest));
    }
    summaries(&mut out, [("TWIN_TWO_LEADERS_MS", &mut two_ms)]);
    writeln!(out, "TWIN_ROUNDS n={rounds} second_led={second_led}").expect("stdout");
}

/// Says `Hello` to each of `replicas` in turn until one answers `Welcome`:
/// the time since `started`. What answered each `Hello` before it is
/// counted by name.
async fn first_welcome(
    replicas: &[Replica],
    started: Instant,
    answers: &mut BTreeMap<String, u64>,
) -> Duration {
    loop {
        for replica in replicas {
            match replica.hello("gateway-a").await {
                Ok(_) => return started.elapsed(),
                Err(answer) => *answers.entry(answer).or_default() += 1,
            }
        }
        assert!(
            started.elapsed() < ROOM,
            "no Welcome in {ROOM:?}: {answers:?}"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
}

/// What a start costs a gateway, at the design's default settings or at
/// the timings the environment names. Over `SMG_REPLICAS_ROUNDS` rounds
/// (default 5) a controller starts alone over the lease in its own
/// process, as one replica runs by default, and then three start over one
/// new lease file. A gateway says `Hello` to each replica in turn every few
/// milliseconds, and its first `Welcome` is timed from the first start;
/// what answered every `Hello` before it is counted by name. A measurement,
/// run by name with `--ignored`; it asserts nothing of the election.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "a measurement for the Numbers table, run by name"]
async fn a_gateways_first_welcome_after_a_start() {
    let mut out = std::io::stdout();
    let settings = replicas::timings(ElectionSettings::default(), &mut out);
    let rounds = rounds("SMG_REPLICAS_ROUNDS", 5);
    let (mut alone_ms, mut shared_ms) = (Vec::new(), Vec::new());
    let mut answers = BTreeMap::new();
    for round in 0..rounds {
        let dir = tempfile::tempdir().expect("a temp dir");
        let lease = dir.path().join("lease");
        for (ids, place, samples) in [
            (&["a"][..], Lease::Own, &mut alone_ms),
            (&["b", "c", "d"][..], Lease::File(&lease), &mut shared_ms),
        ] {
            let started = Instant::now();
            let replicas: Vec<Replica> = ids
                .iter()
                .map(|id| {
                    Replica::start(
                        &format!("replica-{round}-{id}"),
                        place,
                        settings,
                        SETTLE_SECS,
                    )
                    .expect("the replica starts")
                })
                .collect();
            samples.push(millis(
                first_welcome(&replicas, started, &mut answers).await,
            ));
        }
    }
    summaries(
        &mut out,
        [
            ("ALONE_START_TO_WELCOME_MS", &mut alone_ms),
            ("SHARED_START_TO_WELCOME_MS", &mut shared_ms),
        ],
    );
    writeln!(out, "WELCOME_ROUNDS n={rounds} answers_before={answers:?}").expect("stdout");
}
