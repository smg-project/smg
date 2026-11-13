//! The three-process Kubernetes cell (design 10; the step's gate for the
//! `Lease` backend): three controllers, each in its own process, over one
//! `Lease` object on an API server stand-in the parent serves over HTTP
//! (the three calls the backend makes, the resource version's precondition,
//! the server's answers). The holder takes the lease, renews it twice and
//! stalls; two candidates on one tick grid race the take once their sight
//! of the record is a duration old, and the precondition lets exactly one
//! of them write the next term: the other reads what it wrote and follows.
//! The parent reads the three logs and the stand-in's record of holder
//! changes and checks what the fence and the term promise: the holder's
//! fence closed before the taker's first fenced instant, one holder per
//! term, the loser's take never stored, the holder's late renewal refused.
//! The children are this test binary run again with a role, so the client
//! stack, the HTTP round trips and the precondition cross real process
//! boundaries.
#![expect(
    clippy::expect_used,
    reason = "the cell's helpers run inside the test and its child processes; a failure there is the test's failure"
)]

use std::{
    env, fs,
    io::Write,
    net::SocketAddr,
    path::Path,
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

fn holder(url: &str) {
    let mut holder = candidate(url, HOLDER);
    step(&mut holder, HOLDER);
    for _ in 0..2 {
        thread::sleep(settings().renew_period);
        step(&mut holder, HOLDER);
    }
    // The stall: longer than the duration the candidates wait after their
    // first sight of the last renewal, with a grid step and a margin.
    thread::sleep(2 * settings().duration);
    step(&mut holder, HOLDER);
    let released = holder.elector.release().expect("the release");
    log(&format!("{HOLDER} release {released}"));
}

fn taker(url: &str, identity: &str) {
    let mut taker = candidate(url, identity);
    let start = Instant::now();
    while start.elapsed() < 4 * settings().duration {
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
        assert!(
            created.elapsed() < Duration::from_secs(10),
            "no take within 10 s"
        );
        thread::sleep(Duration::from_millis(5));
    }
    let candidate_logs = CANDIDATES.map(|identity| logs.path().join(format!("{identity}.log")));
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
            if tick.starts_with("Acquired { term: 2 }") {
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
