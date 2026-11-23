//! The policy spec's store source end to end (design 4.6): this package's
//! binary started on a spec store, an empty directory, on ephemeral ports,
//! and handed its document by the generated client on the control listener.
//! Before any apply the start reads ready and a decision is shed, declined
//! by the binding. An enforce is on file and runs as generation 1, the read
//! call answers its bytes, and the next one runs as generation 2. The
//! process is then killed and started again on the same directory: it runs
//! generation 2 before any call, under the bytes its caller sent, rolls
//! back to a generation the process before it wrote, and answers each
//! generation on file by its number. A start beside a controller that
//! holds the directory waits for that one to stop and runs its last apply,
//! or exits at its wait's end. A start the store refuses exits, naming
//! why, and so does a start that names no source of its spec.
//!
//! The cases wait on an answer, never on a clock: a wait that outlasts its
//! room fails with the last answer it saw, or names the answer that never
//! came.
#![cfg(unix)]
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
    reason = "the election's record and the lease object's replicas are the replicas cases'"
)]
#[path = "common/replicas.rs"]
mod replicas;
#[path = "common/spec.rs"]
mod spec;

use std::{
    collections::BTreeSet,
    ffi::OsStr,
    fs,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Instant,
};

use replicas::{Lease, Replica, ROOM};
use smg_policy_controller::service::{
    elector::ElectionSettings,
    proto::{
        controller_frame, decide_answer, decision, gateway_frame,
        policy_controller_client::PolicyControllerClient, snapshot_request::Slice, ApplyMode,
        ApplyResult, DecideRequest, Decision, GatewayFrame, Hello, RequestFacts, RollbackRequest,
        ShedReason, SnapshotRequest, SpecStatus,
    },
};
use spec::{
    answers, apply, apply_over, as_operator, control, inside, page, read, until, LEAST_LOAD, PACE,
    RANDOM,
};
use tokio::sync::mpsc;
use tonic::{
    codegen::tokio_stream::wrappers::ReceiverStream, metadata::AsciiMetadataValue, Code, Request,
    Response, Status,
};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// The name of the lock file a store keeps in its directory.
const LOCK: &str = ".lock";

/// What a start writes once when it finds its directory held.
const WAITS: &str = "spec store held by another controller: waiting for it to stop";

/// `LEAST_LOAD` under a weight the validator refuses.
fn unweighed() -> String {
    LEAST_LOAD.replace("[least-load]", "[{ least-load: { weight: -1.0 } }]")
}

/// A replica that leads itself, its lease in its own process, its gateway
/// and control listeners taking the name a call states, on the spec store
/// at `dir`.
fn start(id: &str, dir: &Path) -> Replica {
    let flags = [
        OsStr::new("--identity-as-told"),
        OsStr::new("gateway,control"),
        OsStr::new("--spec-store"),
        dir.as_os_str(),
    ];
    Replica::start_with(id, Lease::Own, ElectionSettings::default(), 0, &flags)
        .expect("the replica starts")
}

/// `frames` as a stream `gateway-a` opens.
fn as_gateway<T>(frames: mpsc::Receiver<T>) -> Request<ReceiverStream<T>> {
    let mut request = Request::new(ReceiverStream::new(frames));
    request.metadata_mut().insert(
        "x-smg-principal",
        AsciiMetadataValue::from_static("gateway-a"),
    );
    request
}

/// What one `Decide` is answered on the gateway listener: asked as
/// `gateway-a`, behind that gateway's `Hello` on a session held open. Each
/// of its five waits on the controller ends inside the room.
async fn decide(replica: &Replica) -> Option<decide_answer::Answer> {
    let dial = PolicyControllerClient::connect(format!("http://{}", replica.gateway));
    let mut client = inside("the gateway listener's connection", dial)
        .await
        .expect("the gateway listener takes a connection");
    let (frames, sent) = mpsc::channel(1);
    let mut session = inside("the session's open", client.session(as_gateway(sent)))
        .await
        .expect("a session opens")
        .into_inner();
    let hello = Hello {
        gateway_id: "gateway-a".to_owned(),
        protocol_min: 1,
        protocol_max: 1,
        ..Hello::default()
    };
    frames
        .send(GatewayFrame {
            seq: 0,
            term_floor: 0,
            frame: Some(gateway_frame::Frame::Hello(hello)),
        })
        .await
        .expect("the session takes the Hello");
    let welcome = inside("the Welcome", session.message())
        .await
        .expect("a frame, not a status");
    assert!(
        matches!(
            welcome.as_ref().and_then(|frame| frame.frame.as_ref()),
            Some(controller_frame::Frame::Welcome(_))
        ),
        "{welcome:?}"
    );
    let (requests, sent) = mpsc::channel(1);
    let mut answers = inside("the decide stream's open", client.decide(as_gateway(sent)))
        .await
        .expect("a present gateway's stream opens")
        .into_inner();
    requests
        .send(DecideRequest {
            request_id: vec![1],
            facts: Some(RequestFacts {
                model: "model-a".to_owned(),
                budget_micros: 5_000_000,
                ..RequestFacts::default()
            }),
            constraints: None,
            again: false,
            ..Default::default()
        })
        .await
        .expect("the stream takes the request");
    inside("the decision", answers.message())
        .await
        .expect("an answer, not a status")
        .expect("an answer, not the stream's end")
        .answer
}

/// One `RollbackPolicySpec` call on the control listener, as `operator-a`.
async fn rollback(replica: &Replica, generation: u64) -> Result<ApplyResult, Status> {
    let mut client = control(replica).await?;
    let call = client.rollback_policy_spec(as_operator(RollbackRequest { generation }));
    inside("the rollback call's answer", call)
        .await
        .map(Response::into_inner)
}

/// The spec slice of the operators' snapshot on the control listener, as
/// `operator-a`.
async fn status(replica: &Replica) -> Result<SpecStatus, Status> {
    let mut client = control(replica).await?;
    let request = SnapshotRequest {
        slices: vec![Slice::Spec.into()],
        ..SnapshotRequest::default()
    };
    let call = client.get_snapshot(as_operator(request));
    let answer = inside("the snapshot call's answer", call).await?;
    Ok(answer.into_inner().spec.expect("the spec slice"))
}

/// The names in the store's directory, the store's own lock file apart.
fn on_file(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .expect("the directory")
        .map(|entry| entry.expect("an entry").file_name())
        .map(|name| name.to_string_lossy().into_owned())
        .filter(|name| name != LOCK)
        .collect()
}

/// The name generation `number` is on file under, as YAML.
fn name(number: u64) -> String {
    format!("{number:020}.yaml")
}

/// The binary started with `flags` on ephemeral ports, to its exit: its
/// code and its log. A start still running at the room's end is stopped,
/// and has no code.
fn refused(flags: &[&OsStr]) -> (Option<i32>, String) {
    let mut child = Command::new(BIN)
        .args(["--controller-id", "store-refused"])
        .args(["--runtime-worker-threads", "1"])
        .args(["--gateway-listen", "127.0.0.1:0"])
        .args(["--control-listen", "127.0.0.1:0"])
        .args(["--replication-listen", "127.0.0.1:0"])
        .args(["--admin-listen", "127.0.0.1:0"])
        .args(flags)
        .env_remove("RUST_LOG")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("the binary runs");
    let started = Instant::now();
    while child.try_wait().expect("the child's state").is_none() {
        if started.elapsed() > ROOM {
            child.kill().expect("the child is stopped");
            break;
        }
        thread::sleep(PACE);
    }
    let output = child.wait_with_output().expect("the child's output");
    let log = String::from_utf8_lossy(&[output.stdout, output.stderr].concat()).into_owned();
    (output.status.code(), log)
}

/// A replica on an empty spec store serves the spec calls with no
/// generation standing, and a dry run there writes no file. An enforce is
/// on file and runs as generation 1, with the feed it needs in its answer;
/// the read call answers its bytes and the series follow. The guard admits
/// the apply that read the standing digest and refuses the next one that
/// names it. A document that does not run, and one whose line is not the
/// next number, are refused by their field with the running generation
/// kept and nothing more on file. Killed and started again on the same
/// directory, the binary runs the newest generation on file before any
/// call, and its next apply is the next number. A rollback there runs a
/// generation the first process wrote as the next number, and the read call
/// answers each generation on file by its number. The snapshot's spec slice
/// names the standing generation and the ones a rollback can name.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_enforce_on_a_spec_store_runs_and_a_start_runs_it_again() {
    let dir = tempfile::tempdir().expect("a directory");
    let mut replica = start("store-0", dir.path());
    // An empty store is a source: the start reads ready before any apply,
    // and no generation stands.
    let ready = page(&replica, "/readiness", None).await;
    assert!(ready.contains(r#""status":"ready""#), "{ready}");
    let nothing = read(&replica, 0).await.expect_err("nothing stands");
    assert_eq!(nothing.code(), Code::NotFound, "{nothing:?}");
    // No worker serves the model, so a decision there is made over no
    // list: a shed for want of endpoints under the stage `none`, whatever
    // document stands, never a pick by a document nobody wrote.
    match decide(&replica).await {
        Some(decide_answer::Answer::Decision(Decision {
            outcome: Some(decision::Outcome::Shed(shed)),
            ..
        })) => {
            assert_eq!(shed.reason(), ShedReason::NoEndpoints, "{shed:?}");
            assert_eq!(shed.emptied_by, "none", "{shed:?}");
        }
        other => panic!("a shed answers a decision before the first apply: {other:?}"),
    }

    // A write is the leader's: the replica answers once it leads its lease.
    let led = until(
        || apply(&replica, ApplyMode::DryRun, LEAST_LOAD),
        Result::is_ok,
    );
    let would = led.await.expect("the leader answers").expect("a verdict");
    assert!(would.accepted && !would.read_only, "{would:?}");
    assert_eq!(would.generation, 1);
    assert_eq!(on_file(dir.path()), BTreeSet::new());
    // The snapshot's spec slice says the same of an empty store: no
    // generation, no digest, nothing a rollback can name.
    let empty = status(&replica).await.expect("the leader's status");
    assert_eq!(
        (
            empty.generation,
            empty.digest.as_str(),
            empty.source.as_str()
        ),
        (0, "", "api")
    );
    assert!(empty.retained.is_empty(), "{empty:?}");

    let first = apply(&replica, ApplyMode::Enforce, LEAST_LOAD)
        .await
        .expect("a verdict");
    assert!(first.accepted && !first.read_only, "{first:?}");
    assert_eq!((first.generation, &first.digest), (1, &would.digest));
    let feed = first.needs.expect("the needs diff");
    assert!(feed.before.expect("the feed before").models.is_empty());
    let after = feed.after.expect("the feed after");
    assert!(
        !after.models.is_empty() && after.models.iter().all(|row| row.loads),
        "{after:?}"
    );
    let stands = answers(&replica, 1).await;
    assert_eq!(stands.document, LEAST_LOAD.as_bytes());
    assert_eq!(
        (stands.format.as_str(), &stands.digest),
        ("yaml", &first.digest)
    );
    assert_eq!(on_file(dir.path()), BTreeSet::from([name(1)]));
    let kept = fs::read(dir.path().join(name(1))).expect("the generation's file");
    assert_eq!(kept, LEAST_LOAD.as_bytes());
    page(&replica, "/metrics", Some("smg_policy_spec_generation 1")).await;

    // The guard: the digest read as standing admits the apply, and refuses
    // the next one that names it, by the generation that stands by then.
    let second = apply_over(&replica, ApplyMode::Enforce, RANDOM, &first.digest)
        .await
        .expect("a verdict");
    assert!(second.accepted, "{second:?}");
    assert_eq!(second.generation, 2);
    let late = apply_over(&replica, ApplyMode::Enforce, LEAST_LOAD, &first.digest)
        .await
        .expect_err("another apply landed first");
    assert_eq!(late.code(), Code::FailedPrecondition, "{late:?}");
    assert_eq!(
        late.message(),
        format!(
            "expected_digest: generation 2 stands with digest {}",
            second.digest
        )
    );

    // A document that does not run, and a line that is not the next number:
    // each is refused by its field, and counted.
    let numbered = format!("generation: 41\n{LEAST_LOAD}");
    for (document, field) in [
        (unweighed(), "profiles.default.score[0]"),
        (numbered, "generation"),
    ] {
        let refused = apply(&replica, ApplyMode::Enforce, &document)
            .await
            .expect("a verdict");
        assert!(!refused.accepted && !refused.read_only, "{refused:?}");
        assert_eq!((refused.generation, &refused.digest), (2, &second.digest));
        let [issue] = refused.issues.as_slice() else {
            panic!("one issue: {refused:?}")
        };
        assert!(issue.field_path.starts_with(field), "{issue:?}");
    }
    assert_eq!(on_file(dir.path()), BTreeSet::from([name(1), name(2)]));
    let metrics = page(&replica, "/metrics", Some("smg_policy_spec_generation 2")).await;
    for series in [
        "smg_policy_spec_apply_total{result=\"applied\"} 2",
        "smg_policy_spec_apply_total{result=\"rejected\"} 2",
    ] {
        assert!(metrics.lines().any(|line| line == series), "{metrics}");
    }

    // Killed, and started again on the same directory: the newest
    // generation on file runs before any call, under its caller's bytes.
    replica.kill().expect("the replica is killed");
    replica.exited().expect("the replica is gone");
    drop(replica);
    let again = start("store-1", dir.path());
    let stands = answers(&again, 2).await;
    assert_eq!(stands.document, RANDOM.as_bytes());
    assert_eq!(stands.digest, second.digest);
    page(&again, "/metrics", Some("smg_policy_spec_generation 2")).await;
    let led = until(
        || apply(&again, ApplyMode::Enforce, LEAST_LOAD),
        Result::is_ok,
    );
    let third = led.await.expect("the leader answers").expect("a verdict");
    assert!(third.accepted, "{third:?}");
    assert_eq!(third.generation, 3);
    let three = BTreeSet::from([name(1), name(2), name(3)]);
    assert_eq!(on_file(dir.path()), three);

    // A rollback runs a generation on file again as the next number, and
    // the read call answers each generation on file by its number.
    let back = rollback(&again, 2).await.expect("a verdict");
    assert!(back.accepted && !back.read_only, "{back:?}");
    assert_eq!((back.generation, &back.digest), (4, &second.digest));
    for (generation, document) in [(1, LEAST_LOAD), (2, RANDOM), (3, LEAST_LOAD), (4, RANDOM)] {
        let held = read(&again, generation).await.expect("on file");
        assert_eq!(
            (held.generation, held.document.as_slice()),
            (generation, document.as_bytes())
        );
    }
    let kept = fs::read(dir.path().join(name(4))).expect("the generation's file");
    assert_eq!(kept, RANDOM.as_bytes());
    let gone = rollback(&again, 9).await.expect_err("not on file");
    assert_eq!(gone.code(), Code::NotFound, "{gone:?}");
    let four = BTreeSet::from([name(1), name(2), name(3), name(4)]);
    assert_eq!(on_file(dir.path()), four);
    // The snapshot's spec slice names what stands and what a rollback can
    // name: the list this start read from the directory, then its two
    // writes.
    let told = status(&again).await.expect("the leader's status");
    assert_eq!(
        (told.generation, &told.digest, told.source.as_str()),
        (4, &second.digest, "api")
    );
    assert_eq!(told.retained, [1, 2, 3]);
}

/// A start on a directory another controller holds waits for that one to
/// stop, as a pod replaced outside an upgrade starts beside the one it
/// replaces. One still held at the end of its wait exits, naming why, and
/// nothing on file moves. One whose holder stops inside the wait reads the
/// directory only then: it runs what the holder applied while it waited,
/// and its own apply is the next number.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_start_on_a_held_spec_store_waits_for_its_holder_and_runs_its_last_apply() {
    let dir = tempfile::tempdir().expect("a directory");
    let mut holder = start("store-holder", dir.path());
    let led = until(
        || apply(&holder, ApplyMode::Enforce, LEAST_LOAD),
        Result::is_ok,
    );
    let first = led.await.expect("the leader answers").expect("a verdict");
    assert_eq!(first.generation, 1, "{first:?}");
    assert!(dir.path().join(LOCK).is_file(), "the holder made its lock");

    // A start whose wait is one second: held at its end, it exits.
    let mut short = vec![OsStr::new("--spec-store"), dir.path().as_os_str()];
    short.extend(["--shutdown-settle-secs", "0"].map(OsStr::new));
    short.extend(["--shutdown-grace-period-secs", "1"].map(OsStr::new));
    let (code, log) = refused(&short);
    assert_eq!(code, Some(1), "{log}");
    assert!(log.contains("start refused"), "{log}");
    assert!(
        log.contains("another controller holds the directory, waited on for 1s"),
        "{log}"
    );
    assert_eq!(on_file(dir.path()), BTreeSet::from([name(1)]));

    // A start whose wait is thirty seconds, beside the holder: it says
    // once that it waits, having read no name yet.
    let flags = [
        OsStr::new("--identity-as-told"),
        OsStr::new("gateway,control"),
        OsStr::new("--spec-store"),
        dir.path().as_os_str(),
    ];
    let settings = ElectionSettings::default();
    let waiter = Replica::start_with("store-waiter", Lease::Own, settings, 15, &flags)
        .expect("the replica starts");
    assert!(
        waiter.logs(|replica| replica.logged(WAITS)).await,
        "{}",
        waiter.log()
    );
    // The holder applies once more while the other waits, then stops.
    let second = apply(&holder, ApplyMode::Enforce, RANDOM)
        .await
        .expect("a verdict");
    assert_eq!(second.generation, 2, "{second:?}");
    holder.terminate().expect("SIGTERM");
    let status = holder.exited().expect("the holder exits");
    assert!(status.success(), "read {status}:\n{}", holder.log());

    // The waiter holds the directory now. It runs the holder's last apply,
    // and its own is the next number.
    let stands = answers(&waiter, 2).await;
    assert_eq!(stands.document, RANDOM.as_bytes());
    assert_eq!(stands.digest, second.digest);
    let led = until(
        || apply(&waiter, ApplyMode::Enforce, LEAST_LOAD),
        Result::is_ok,
    );
    let third = led.await.expect("the leader answers").expect("a verdict");
    assert_eq!(third.generation, 3, "{third:?}");
    let three = BTreeSet::from([name(1), name(2), name(3)]);
    assert_eq!(on_file(dir.path()), three);
}

/// A start that names no source of the policy spec is refused by the
/// command line, which names the two flags it takes one of. Nothing is
/// served.
#[test]
fn a_start_that_names_no_spec_source_is_refused_naming_both_flags() {
    let (code, log) = refused(&[]);
    assert_eq!(code, Some(2), "{log}");
    assert!(log.contains("required"), "{log}");
    assert!(
        log.contains("--spec-file") && log.contains("--spec-store"),
        "{log}"
    );
}

/// A start the store refuses exits with the refusal in its log: beside a
/// spec file, by the two flags; beside a lease another replica can take; on
/// a path that leads to no directory, to nothing or through a file; and on
/// a newest generation that does not run, by its number and its field.
/// Nothing is served.
#[test]
fn a_start_a_spec_store_refuses_exits_naming_why() {
    let dir = tempfile::tempdir().expect("a directory");
    let store = dir.path().join("store");
    fs::create_dir(&store).expect("the store's directory");
    let file = dir.path().join("policy.yaml");
    fs::write(&file, LEAST_LOAD).expect("the file");
    let (lease, nowhere) = (dir.path().join("lease"), dir.path().join("nowhere"));
    let on_store = |flags: &[&OsStr]| {
        let mut all = vec![OsStr::new("--spec-store"), store.as_os_str()];
        all.extend_from_slice(flags);
        refused(&all)
    };

    let (code, log) = on_store(&[OsStr::new("--spec-file"), file.as_os_str()]);
    assert_eq!(code, Some(2), "{log}");
    assert!(
        log.contains("--spec-store") && log.contains("cannot be used with '--spec-file"),
        "{log}"
    );

    let (code, log) = on_store(&[OsStr::new("--lease-file"), lease.as_os_str()]);
    assert_eq!(code, Some(1), "{log}");
    assert!(log.contains("start refused"), "{log}");
    assert!(
        log.contains(&store.display().to_string()),
        "the directory is named: {log}"
    );
    assert!(
        log.contains("another replica can take this controller's lease"),
        "{log}"
    );

    let through = file.join("store");
    for path in [&nowhere, &through] {
        let (code, log) = refused(&[OsStr::new("--spec-store"), path.as_os_str()]);
        assert_eq!(code, Some(1), "{log}");
        assert!(log.contains("the path leads to no directory"), "{log}");
    }

    fs::write(store.join(name(1)), unweighed()).expect("a generation on file");
    let (code, log) = on_store(&[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("generation 1") && log.contains("profiles.default.score[0]"),
        "the generation and its field are named: {log}"
    );
}

/// A start on a spec store named through a link says the directory it
/// holds: its line names the path as resolved, the one the line of a wait
/// names.
#[cfg(unix)]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_start_through_a_link_names_the_directory_it_holds() {
    use std::os::unix::fs::symlink;

    let base = tempfile::tempdir().expect("a directory");
    let (dir, link) = (base.path().join("held"), base.path().join("link"));
    fs::create_dir(&dir).expect("the directory");
    symlink(&dir, &link).expect("the link");
    let replica = start("store-link", &link);
    let said = replica.logs(|replica| replica.logged("spec store")).await;
    let log = replica.log();
    assert!(said, "{log}");
    let named = log
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|entry| entry["fields"]["message"] == "spec store")
        .and_then(|entry| entry["fields"]["path"].as_str().map(str::to_owned));
    let held = fs::canonicalize(&dir).expect("its path");
    assert_eq!(named.as_deref(), held.to_str(), "{log}");
}
