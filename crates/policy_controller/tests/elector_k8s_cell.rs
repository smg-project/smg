//! The three-process Kubernetes cell (design 10; the step's gate for the
//! `Lease` backend): three controllers, each in its own process, over one
//! `Lease` object on an API server stand-in the parent serves over HTTP
//! (the three calls the backend makes, the resource version's precondition,
//! the server's answers). The holder takes the lease, renews it twice and
//! stalls until a candidate took it; two candidates on one tick grid race
//! the take once their sight of the record is a duration old, and the
//! precondition lets exactly one of them write the next term: the other
//! reads what it wrote and follows.
//! The parent reads the three logs and the stand-in's record of holder
//! changes and checks what the fence and the term promise: the holder's
//! fence closed before the taker's first fenced instant, one holder per
//! term, the loser's take never stored, the holder's late renewal refused.
//! The children are this test binary run again with a role, so the client
//! stack, the HTTP round trips and the precondition cross real process
//! boundaries.
//!
//! With the `controller` feature four more cells run the package's binary
//! over the same stand-in: two replicas named the `Lease` by flag, one
//! leader, a hand-over at its stop, the stopped process gone only after
//! its release was stored; a release the stand-in refuses once, written
//! again a renew period later; a release stored while the stopped leader
//! still answers, ahead of its settle window; a read the stand-in never
//! answers, failed at the renew deadline.
#![expect(
    clippy::expect_used,
    reason = "the cell's helpers run inside the test and its child processes; a failure there is the test's failure"
)]

use std::{
    env, fs,
    io::Write,
    net::SocketAddr,
    path::{Path, PathBuf},
    process::{Child, Command, Output, Stdio},
    sync::{mpsc, Arc},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use k8s_openapi::api::coordination::v1::Lease;
use kube::{core::Status, Client, Config};
use parking_lot::Mutex;
use smg_policy_controller::elector::{
    ElectionSettings, Elector, KubernetesLease, Tick, TERM_ANNOTATION,
};

const ROLE: &str = "SMG_ELECTOR_K8S_CELL_ROLE";
const URL: &str = "SMG_ELECTOR_K8S_CELL_URL";
const LOG: &str = "SMG_ELECTOR_K8S_CELL_LOG";
const TEST: &str = "three_processes_over_one_lease_object_hand_over_once_per_term";
const NAMESPACE: &str = "policy";
const NAME: &str = "controller";
const HOLDER: &str = "a@cell:1";
const CANDIDATES: [&str; 2] = ["b@cell:2", "c@cell:3"];
/// The tick a candidate logs when it takes the lease from the holder.
const TAKE: &str = "Acquired { term: 2 }";
/// How long a wait of the cell lasts before the cell is broken. Every wait
/// ends at its event (the first take, the holder's stall at the second, a
/// candidate's loop at its own take or at the winner's term), so a process
/// that starts late moves the events and never meets this bound.
const BROKEN: Duration = Duration::from_secs(10);

/// The design's order between the settings at a third of its size, with the
/// whole second the `Lease` stores.
fn settings() -> ElectionSettings {
    ElectionSettings {
        duration: Duration::from_secs(1),
        renew_period: Duration::from_millis(250),
        renew_deadline: Duration::from_millis(500),
    }
}

fn wall_micros() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|since| u64::try_from(since.as_micros()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

fn deadline_micros() -> u64 {
    u64::try_from(settings().renew_deadline.as_micros()).expect("a small number")
}

// The stand-in: one `Lease` object, its resource version, and the holder
// changes it stored.

#[derive(Default)]
struct Store {
    object: Option<Lease>,
    version: u64,
    /// Every stored change of holder or term as `holder@term`, in order.
    changes: Vec<String>,
    /// How long a release waits before it is stored: an API server slower
    /// than the releasing process's way out.
    release_delay: Duration,
    /// How many releases are refused before one is stored: an API server
    /// away for a moment.
    release_refusals: u32,
    /// When each release arrived, refused or stored.
    releases: Vec<Instant>,
    /// Whether a read is never answered: a round trip that stalls.
    reads_stall: bool,
}

type Shared = Arc<Mutex<Store>>;

fn failure(code: StatusCode, reason: &str) -> Response {
    let body = Status::failure(reason, reason).with_code(code.as_u16());
    (code, Json(body)).into_response()
}

fn store(store: &mut Store, mut object: Lease) -> Lease {
    store.version += 1;
    object.metadata.resource_version = Some(store.version.to_string());
    let holder = object
        .spec
        .as_ref()
        .and_then(|spec| spec.holder_identity.clone())
        .unwrap_or_default();
    let term = object
        .metadata
        .annotations
        .as_ref()
        .and_then(|annotations| annotations.get(TERM_ANNOTATION).cloned())
        .unwrap_or_default();
    let change = format!("{holder}@{term}");
    if store.changes.last() != Some(&change) {
        store.changes.push(change);
    }
    store.object = Some(object.clone());
    object
}

async fn get_lease(State(shared): State<Shared>) -> Response {
    let stalls = shared.lock().reads_stall;
    if stalls {
        // The client's own timeout ends the round trip.
        std::future::pending::<()>().await;
    }
    match &shared.lock().object {
        Some(object) => Json(object.clone()).into_response(),
        None => failure(StatusCode::NOT_FOUND, "NotFound"),
    }
}

async fn create_lease(State(shared): State<Shared>, Json(object): Json<Lease>) -> Response {
    let mut guard = shared.lock();
    if guard.object.is_some() {
        return failure(StatusCode::CONFLICT, "AlreadyExists");
    }
    let stored = store(&mut guard, object);
    (StatusCode::CREATED, Json(stored)).into_response()
}

async fn replace_lease(State(shared): State<Shared>, Json(object): Json<Lease>) -> Response {
    let releasing = object
        .spec
        .as_ref()
        .is_none_or(|spec| spec.holder_identity.is_none());
    let delay = if releasing {
        let mut guard = shared.lock();
        guard.releases.push(Instant::now());
        if guard.release_refusals > 0 {
            guard.release_refusals -= 1;
            return failure(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailable");
        }
        guard.release_delay
    } else {
        Duration::ZERO
    };
    if !delay.is_zero() {
        tokio::time::sleep(delay).await;
    }
    let mut guard = shared.lock();
    let Some(current) = &guard.object else {
        return failure(StatusCode::NOT_FOUND, "NotFound");
    };
    if object.metadata.resource_version != current.metadata.resource_version {
        return failure(StatusCode::CONFLICT, "Conflict");
    }
    let stored = store(&mut guard, object);
    Json(stored).into_response()
}

/// Serves the stand-in from a thread of its own on a port of its own; the
/// address once it listens.
fn serve(shared: Shared) -> SocketAddr {
    let (ready, listening) = mpsc::channel();
    thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("a runtime");
        runtime.block_on(async move {
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                .await
                .expect("a port");
            ready
                .send(listener.local_addr().expect("the address"))
                .expect("the parent waits for the address");
            let leases = format!("/apis/coordination.k8s.io/v1/namespaces/{NAMESPACE}/leases");
            let router = Router::new()
                .route(&leases, post(create_lease))
                .route(
                    &format!("{leases}/{NAME}"),
                    get(get_lease).put(replace_lease),
                )
                .with_state(shared);
            axum::serve(listener, router).await.expect("the stand-in");
        });
    });
    listening.recv().expect("the address")
}

// The children.

/// An elector over the stand-in, with the runtime its client blocks on; the
/// elector drops first.
struct Candidate {
    elector: Elector,
    _runtime: tokio::runtime::Runtime,
}

fn candidate(url: &str, identity: &str) -> Candidate {
    let runtime = tokio::runtime::Runtime::new().expect("a runtime");
    let client = runtime.block_on(async {
        Client::try_from(Config::new(url.parse().expect("the stand-in's url"))).expect("a client")
    });
    let backend = KubernetesLease::new(client, NAMESPACE, NAME, runtime.handle().clone());
    let elector = Elector::new(Box::new(backend), identity, settings()).expect("settings");
    Candidate {
        elector,
        _runtime: runtime,
    }
}

/// One line of the child's log, the file the parent named.
fn log(line: &str) {
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(env::var(LOG).expect("the child's log path"))
        .expect("the log");
    writeln!(file, "{line}").expect("a log line");
}

/// One tick, logged as `<identity> <wall before> <wall after> <tick>`: the
/// instant the tick runs on is taken before `before`, so a fence it opens
/// closes no later than `before` plus the renew deadline, and no earlier
/// than `before` itself.
fn step(candidate: &mut Candidate, identity: &str) -> Tick {
    let now = Instant::now();
    let before = wall_micros();
    let tick = candidate.elector.tick(now).expect("a tick");
    log(&format!("{identity} {before} {} {tick:?}", wall_micros()));
    tick
}

/// Sleeps to the next quarter second of wall time, the grid both candidates
/// tick on, so their takes race on the server.
fn sleep_to_grid() {
    let period = 250_000;
    let now = wall_micros();
    let next = (now / period + 1) * period;
    thread::sleep(Duration::from_micros(next - now));
}

/// A candidate's log, beside the holder's in the parent's directory.
fn candidate_log(logs: &Path, identity: &str) -> PathBuf {
    logs.join(format!("{identity}.log"))
}

/// The holder's stall: no tick until a candidate's log holds its take. The
/// stall ends at the take, never at a fixed time, so candidates that start
/// late move its end with them: a holder that woke before the take would
/// renew its own unmoved record and the cell would have no loss to read.
/// Bounded for a broken cell.
fn stall_until_taken() {
    let own = env::var(LOG).expect("the child's log path");
    let logs = Path::new(&own).parent().expect("the logs' directory");
    let stalled = Instant::now();
    while !CANDIDATES.iter().any(|identity| {
        fs::read_to_string(candidate_log(logs, identity)).is_ok_and(|log| log.contains(TAKE))
    }) {
        assert!(
            stalled.elapsed() < BROKEN,
            "no candidate took the lease within {BROKEN:?} of the holder's last renewal"
        );
        thread::sleep(Duration::from_millis(10));
    }
}

fn holder(url: &str) {
    let mut holder = candidate(url, HOLDER);
    step(&mut holder, HOLDER);
    for _ in 0..2 {
        thread::sleep(settings().renew_period);
        step(&mut holder, HOLDER);
    }
    stall_until_taken();
    step(&mut holder, HOLDER);
    let released = holder.elector.release().expect("the release");
    log(&format!("{HOLDER} release {released}"));
}

/// A candidate ticks on the grid until it took the lease or follows the one
/// that did: its loop ends at that event, however late it started.
fn taker(url: &str, identity: &str) {
    let mut taker = candidate(url, identity);
    let start = Instant::now();
    while start.elapsed() < BROKEN {
        sleep_to_grid();
        match step(&mut taker, identity) {
            Tick::Acquired { .. } => {
                for _ in 0..2 {
                    thread::sleep(settings().renew_period);
                    step(&mut taker, identity);
                }
                return;
            }
            Tick::Following { term: 2, .. } => return,
            _ => {}
        }
    }
}

// The parent.

fn spawn(url: &str, role: &str, log: &Path) -> Child {
    Command::new(env::current_exe().expect("this binary"))
        .args([TEST, "--exact", "--nocapture", "--test-threads", "1"])
        .env(ROLE, role)
        .env(URL, url)
        .env(LOG, log)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .expect("a child")
}

fn finished(output: &Output, identity: &str) {
    assert!(
        output.status.success(),
        "{identity}: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// The logged ticks of one child: (wall before, wall after, the tick's text).
fn ticks(log: &Path, identity: &str) -> Vec<(u64, u64, String)> {
    fs::read_to_string(log)
        .expect("the child's log")
        .lines()
        .filter_map(|line| {
            let rest = line.strip_prefix(identity)?.strip_prefix(' ')?;
            let mut words = rest.splitn(3, ' ');
            let before = words.next()?.parse().ok()?;
            let after = words.next()?.parse().ok()?;
            Some((before, after, words.next()?.to_string()))
        })
        .collect()
}

#[test]
fn three_processes_over_one_lease_object_hand_over_once_per_term() {
    if let Ok(role) = env::var(ROLE) {
        let url = env::var(URL).expect("the stand-in's url");
        if role == "holder" {
            holder(&url);
        } else {
            taker(&url, &role);
        }
        return;
    }

    let shared = Shared::default();
    let url = format!("http://{}", serve(Arc::clone(&shared)));
    let logs = tempfile::tempdir().expect("a directory for the logs");
    let holder_log = logs.path().join("holder.log");
    let holder = spawn(&url, "holder", &holder_log);
    // The candidates start once the lease exists, so their first sight is
    // of a held record.
    let created = Instant::now();
    while shared.lock().object.is_none() {
        assert!(created.elapsed() < BROKEN, "no take within {BROKEN:?}");
        thread::sleep(Duration::from_millis(5));
    }
    let candidate_logs = CANDIDATES.map(|identity| candidate_log(logs.path(), identity));
    let candidates = [0, 1].map(|i| spawn(&url, CANDIDATES[i], &candidate_logs[i]));
    finished(
        &holder.wait_with_output().expect("the holder's exit"),
        HOLDER,
    );
    for (child, identity) in candidates.into_iter().zip(CANDIDATES) {
        finished(
            &child.wait_with_output().expect("a candidate's exit"),
            identity,
        );
    }

    let holder_ticks = ticks(&holder_log, HOLDER);
    assert!(
        holder_ticks[0].2.starts_with("Acquired { term: 1 }"),
        "{holder_ticks:?}"
    );
    let last_renewal = holder_ticks
        .iter()
        .filter(|(_, _, tick)| tick.starts_with("Renewed { term: 1 }"))
        .map(|(before, _, _)| *before)
        .max()
        .expect("two renewals");
    let lost = holder_ticks
        .iter()
        .find(|(_, _, tick)| tick.starts_with("Lost { term: 1"))
        .expect("the stalled holder wakes to a moved record");

    let mut takes = Vec::new();
    let mut follows = Vec::new();
    for (identity, log) in CANDIDATES.iter().zip(&candidate_logs) {
        for (before, _, tick) in ticks(log, identity) {
            if tick.starts_with(TAKE) {
                takes.push((*identity, before));
            } else if tick.starts_with("Following") && tick.contains("term: 2") {
                follows.push((*identity, before));
            }
        }
    }
    assert_eq!(takes.len(), 1, "one take at term 2: {takes:?}");
    assert_eq!(follows.len(), 1, "the other follows: {follows:?}");
    assert_ne!(takes[0].0, follows[0].0);
    let (winner, taken_before) = takes[0];
    assert!(
        taken_before > last_renewal + deadline_micros(),
        "the holder's fence closed by {} and the take ran after {taken_before}",
        last_renewal + deadline_micros()
    );
    assert!(
        lost.2.contains(&format!("holder: \"{winner}\"")),
        "the holder lost to the winner: {}",
        lost.2
    );
    assert_eq!(
        shared.lock().changes,
        vec![format!("{HOLDER}@1"), format!("{winner}@2")],
        "one holder per term, the loser's take never stored"
    );
}

// The binary over the stand-in.

#[cfg(feature = "controller")]
#[expect(
    dead_code,
    reason = "this binary starts replicas over the Lease object alone; the lease file's case reads the rest"
)]
#[path = "common/replicas.rs"]
mod replicas;

/// Writes at `kubeconfig` a client configuration that names the stand-in at
/// `url` and its namespace: no flag names the object's namespace, as in a pod.
#[cfg(feature = "controller")]
fn client_configuration(kubeconfig: &Path, url: &str) {
    fs::write(
        kubeconfig,
        format!(
            "apiVersion: v1\nkind: Config\ncurrent-context: stand-in\n\
             clusters:\n- name: stand-in\n  cluster:\n    server: {url}\n\
             contexts:\n- name: stand-in\n  context:\n    cluster: stand-in\n    user: stand-in\n    namespace: {NAMESPACE}\n\
             users:\n- name: stand-in\n  user: {{}}\n"
        ),
    )
    .expect("the client configuration");
}

/// The binary over the same stand-in (design 5.3): two replicas named the
/// `Lease` by flag, both given one client configuration that names the
/// stand-in and its namespace, asked who leads by the generated client.
/// One leads and the other names it; the leader is stopped, releases the
/// object and the other takes it under the next term. The stand-in's record
/// of changes ends with that order: the leader, its release, the successor.
/// A leader the host held back past the duration has lost the object by
/// the time it reads its stop: the record then ends with the leader and the
/// successor, and the successor is stopped alone and releases in its turn.
/// The stand-in stores a release late and the replicas close their
/// listeners at the signal, so the stopped leader's way out is shorter than
/// its release: it leaves once the release is stored all the same.
#[cfg(feature = "controller")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn two_binaries_elect_over_the_lease_object_and_hand_over_at_a_stop() {
    use replicas::{Lease as Place, Record, Replica, ROOM};

    let shared = Shared::default();
    let url = format!("http://{}", serve(Arc::clone(&shared)));
    let dir = tempfile::tempdir().expect("a temp dir");
    let kubeconfig = dir.path().join("kubeconfig");
    client_configuration(&kubeconfig, &url);
    let place = Place::Object {
        name: NAME,
        kubeconfig: &kubeconfig,
    };
    shared.lock().release_delay = settings().renew_deadline / 2;
    let start = |id: &str| Replica::start(id, place, settings(), 0).expect("the replica starts");
    let mut replicas = vec![start("replica-a"), start("replica-b")];
    let mut record = Record::default();

    let (leader, term) = record
        .agreed(&mut replicas)
        .await
        .expect("the replicas agree on a leader");
    let first = replicas[leader].identity();
    let holder = shared
        .lock()
        .object
        .as_ref()
        .and_then(|object| object.spec.as_ref())
        .and_then(|spec| spec.holder_identity.clone());
    assert_eq!(holder.as_deref(), Some(first.as_str()));

    let mut stopped = replicas.remove(leader);
    stopped.terminate().expect("SIGTERM");
    let within = settings().duration + settings().renew_period + ROOM;
    let successor = record
        .promoted(&mut replicas, term, within)
        .await
        .expect("the other replica takes the lease");
    assert!(stopped
        .exited()
        .expect("the stopped leader exits")
        .success());
    assert_eq!(successor.replica, replicas[0].id);
    assert_eq!(successor.info.term, term + 1);

    // The stop is read from the stand-in's record of changes: the leader,
    // its release and the successor, in that order.
    let second = replicas[0].identity();
    let taken = format!("{second}@{}", term + 1);
    let changes = shared.lock().changes.clone();
    if changes.ends_with(&[format!("{first}@{term}"), format!("@{term}"), taken.clone()]) {
        assert!(
            stopped.logged("the lease is released"),
            "a stored release the leader never logged:\n{}",
            stopped.log()
        );
    } else {
        // No release was stored: the other replica took the object by its
        // duration. A leader held back past the duration across its signal
        // has lost the object by the time it reads the stop: it writes
        // nothing and logs nothing, and the record of changes cannot tell
        // that stop from one that never releases. The replica left is
        // stopped alone, where nothing but its own release moves the
        // object.
        assert!(
            changes.ends_with(&[format!("{first}@{term}"), taken]),
            "the object's changes: {changes:?}"
        );
        assert!(
            !stopped.logged("the lease is released"),
            "the leader logged a release the stand-in never stored:\n{}",
            stopped.log()
        );
        let (alone, term) = record
            .agreed(&mut replicas)
            .await
            .expect("the replica left leads");
        let mut stopped = replicas.remove(alone);
        stopped.terminate().expect("SIGTERM");
        assert!(stopped
            .exited()
            .expect("the stopped leader exits")
            .success());
        let changes = shared.lock().changes.clone();
        assert!(
            stopped.logged("the lease is released")
                && changes.ends_with(&[format!("{second}@{term}"), format!("@{term}")]),
            "a leader stopped alone releases the object: {changes:?}\n{}",
            stopped.log()
        );
    }
    record.check().expect("the election's promise holds");
}

/// A release the stand-in refuses once (design 5.3): the stopped leader
/// writes it again no sooner than a renew period after the first, the
/// stand-in stores that one, and the other replica takes the object. The
/// row of the Numbers table is printed: the signal to the stored release
/// and to the take, at the cell's timings or the three of
/// `SMG_REPLICAS_TIMINGS_MS`. A leader the host held back past the duration
/// has lost the object by its second release: the other replica took it
/// unreleased, and the replica left is stopped alone and refused once in
/// its turn.
#[cfg(feature = "controller")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_refused_release_is_written_again_a_renew_period_later() {
    use replicas::{Lease as Place, Record, Replica, ROOM};

    let mut out = std::io::stdout();
    let settings = replicas::timings(settings(), &mut out);
    let shared = Shared::default();
    let url = format!("http://{}", serve(Arc::clone(&shared)));
    let dir = tempfile::tempdir().expect("a temp dir");
    let kubeconfig = dir.path().join("kubeconfig");
    client_configuration(&kubeconfig, &url);
    let place = Place::Object {
        name: NAME,
        kubeconfig: &kubeconfig,
    };
    shared.lock().release_refusals = 1;
    let start = |id: &str| Replica::start(id, place, settings, 0).expect("the replica starts");
    let mut replicas = vec![start("replica-a"), start("replica-b")];
    let mut record = Record::default();
    // What a stop the stand-in refused once leaves: the refusal in the
    // stopped leader's log, and two releases at the stand-in after the
    // `earlier` ones, the second no sooner than a renew period after the
    // signal: the first began after the signal and the second a period
    // after the first began. A hold of the host only makes the second
    // later: the floor needs no room.
    let refused_then_stored = |stopped: &Replica, earlier: usize, signalled: Instant| {
        assert!(
            stopped.logged("the release failed"),
            "the stand-in refused the first release:\n{}",
            stopped.log()
        );
        let releases = shared.lock().releases[earlier..].to_vec();
        assert_eq!(releases.len(), 2, "one refused, one stored");
        let again = releases[1].duration_since(signalled);
        assert!(
            again >= settings.renew_period,
            "the second release came {again:?} after the signal"
        );
    };
    let (leader, term) = record
        .agreed(&mut replicas)
        .await
        .expect("the replicas agree on a leader");
    let first = replicas[leader].identity();

    let mut stopped = replicas.remove(leader);
    // Taken before the signal: no release began sooner.
    let signalled = Instant::now();
    stopped.terminate().expect("SIGTERM");
    let within = settings.duration + settings.renew_period + ROOM;
    record
        .promoted(&mut replicas, term, within)
        .await
        .expect("the other replica takes the lease");
    assert!(stopped
        .exited()
        .expect("the stopped leader exits")
        .success());
    // The stop is read from the stand-in's record of changes: the leader,
    // its release and the successor, in that order.
    let second = replicas[0].identity();
    let taken = format!("{second}@{}", term + 1);
    let changes = shared.lock().changes.clone();
    if changes.ends_with(&[format!("{first}@{term}"), format!("@{term}"), taken.clone()]) {
        assert!(
            stopped.logged("the lease is released"),
            "a stored release the leader never logged:\n{}",
            stopped.log()
        );
        refused_then_stored(&stopped, 0, signalled);

        // The successor lives on: its take's line reaches the log a
        // moment after its answer turned.
        assert!(
            replicas[0]
                .logs(|successor| successor.logged("leading"))
                .await,
            "the successor logged no take:\n{}",
            replicas[0].log()
        );
        let stamp = |replica: &Replica, message: &str| {
            replica
                .logged_at_micros(message)
                .expect("the line is in the log")
        };
        let signal = stamp(&stopped, "SIGTERM received");
        writeln!(
            out,
            "REFUSED_RELEASE signal_to_release_ms={} signal_to_take_ms={}",
            stamp(&stopped, "the lease is released").saturating_sub(signal) / 1000,
            stamp(&replicas[0], "leading").saturating_sub(signal) / 1000
        )
        .expect("stdout");
    } else {
        // No release was stored: the other replica took the object by its
        // duration. A leader held back past the duration, before its signal
        // or between its two releases, has lost the object by the time it
        // releases: it writes nothing and logs nothing, and the record of
        // changes cannot tell that stop from one that never releases. The
        // replica left is stopped alone, where nothing but its own release
        // moves the object, and the stand-in refuses its first release.
        assert!(
            changes.ends_with(&[format!("{first}@{term}"), taken]),
            "the object's changes: {changes:?}"
        );
        assert!(
            !stopped.logged("the lease is released"),
            "the leader logged a release the stand-in never stored:\n{}",
            stopped.log()
        );
        let (alone, term) = record
            .agreed(&mut replicas)
            .await
            .expect("the replica left leads");
        let mut stopped = replicas.remove(alone);
        let earlier = {
            let mut store = shared.lock();
            store.release_refusals = 1;
            store.releases.len()
        };
        let signalled = Instant::now();
        stopped.terminate().expect("SIGTERM");
        assert!(stopped
            .exited()
            .expect("the stopped leader exits")
            .success());
        let changes = shared.lock().changes.clone();
        assert!(
            stopped.logged("the lease is released")
                && changes.ends_with(&[format!("{second}@{term}"), format!("@{term}")]),
            "a leader stopped alone releases the object: {changes:?}\n{}",
            stopped.log()
        );
        refused_then_stored(&stopped, earlier, signalled);
    }
    record.check().expect("the election's promise holds");
}

/// A settle window longer than the cell below waits for anything: a stop
/// that waited it out before its release would store none in that time.
#[cfg(feature = "controller")]
const LONG_SETTLE_SECS: u64 = 2 * replicas::ROOM.as_secs();

/// The release is at the signal, ahead of the settle window (design 5.3):
/// a leader stopped under a settle window longer than the cell waits has
/// its release stored, and then still answers `Leader`, as a replica that
/// does not lead. The cell holds the stop to that order and to no clock: a
/// slow store delays the cell and no more.
#[cfg(feature = "controller")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_stopped_leader_releases_while_its_listeners_still_answer() {
    use replicas::{Lease as Place, Record, Replica, ROOM};

    let shared = Shared::default();
    let url = format!("http://{}", serve(Arc::clone(&shared)));
    let dir = tempfile::tempdir().expect("a temp dir");
    let kubeconfig = dir.path().join("kubeconfig");
    client_configuration(&kubeconfig, &url);
    let place = Place::Object {
        name: NAME,
        kubeconfig: &kubeconfig,
    };
    let mut lone = [
        Replica::start("replica-a", place, settings(), LONG_SETTLE_SECS)
            .expect("the replica starts"),
    ];
    let mut record = Record::default();
    record
        .agreed(&mut lone)
        .await
        .expect("the lone replica leads");

    lone[0].terminate().expect("SIGTERM");
    let signalled = Instant::now();
    let released = || {
        let last = shared.lock().changes.last().cloned();
        last.is_some_and(|change| change.starts_with('@'))
    };
    while !released() {
        assert!(
            signalled.elapsed() < ROOM,
            "no release was stored in {ROOM:?} of the signal:\n{}",
            lone[0].log()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    let answer = loop {
        if let Some(answer) = record.sample(&mut lone).await.pop() {
            break answer;
        }
        assert!(
            signalled.elapsed() < ROOM,
            "the stopped leader answers through its settle window:\n{}",
            lone[0].log()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    };
    assert!(
        !answer.info.is_leader,
        "the stopped leader answers as a replica that does not lead: {answer:?}"
    );
    record.check().expect("the election's promise holds");
}

/// The line a failed election step logs, once a streak.
#[cfg(feature = "controller")]
const FAILED_STEP: &str =
    "an election step failed; the fence closes at the renew deadline unless a later step lands";

/// A read the stand-in never answers (design 5.3): the lease client's
/// timeouts are the renew deadline, so the replica's look fails there and
/// its thread goes on to the next, where the client's own would hold the
/// thread for minutes.
#[cfg(feature = "controller")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_look_the_lease_store_never_answers_fails_at_the_renew_deadline() {
    use replicas::{Lease as Place, Replica, ROOM};

    let shared = Shared::default();
    shared.lock().reads_stall = true;
    let url = format!("http://{}", serve(Arc::clone(&shared)));
    let dir = tempfile::tempdir().expect("a temp dir");
    let kubeconfig = dir.path().join("kubeconfig");
    client_configuration(&kubeconfig, &url);
    let place = Place::Object {
        name: NAME,
        kubeconfig: &kubeconfig,
    };
    let replica = Replica::start("replica-a", place, settings(), 0).expect("the replica starts");
    let within = settings().renew_deadline + ROOM;
    let started = Instant::now();
    while !replica.logged(FAILED_STEP) {
        assert!(
            started.elapsed() < within,
            "no look failed in {within:?}:\n{}",
            replica.log()
        );
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}
