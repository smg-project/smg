//! The policy spec's store source end to end (design 4.6): this package's
//! binary started on a spec store, an empty directory, on ephemeral ports,
//! and handed its document by the generated client on the control listener.
//! An enforce is on file and runs as generation 1, the read call answers its
//! bytes, and the next one runs as generation 2. The process is then killed
//! and started again on the same directory: it runs generation 2 before any
//! call, under the bytes its caller sent. A start the store refuses exits,
//! naming why.
//!
//! The cases wait on an answer, never on a clock: a wait that outlasts its
//! room fails with the last answer it saw.
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
use smg_policy_controller::{elector::ElectionSettings, proto::ApplyMode};
use spec::{answers, apply, apply_over, page, read, until, LEAST_LOAD, PACE, RANDOM};
use tonic::Code;

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// `LEAST_LOAD` under a weight the validator refuses.
fn unweighed() -> String {
    LEAST_LOAD.replace("[least-load]", "[{ least-load: { weight: -1.0 } }]")
}

/// A replica that leads itself, its lease in its own process, its control
/// listener taking the name a call states, on the spec store at `dir`.
fn start(id: &str, dir: &Path) -> Replica {
    let flags = [
        OsStr::new("--identity-as-told"),
        OsStr::new("control"),
        OsStr::new("--spec-store"),
        dir.as_os_str(),
    ];
    Replica::start_with(id, Lease::Own, ElectionSettings::default(), 0, &flags)
        .expect("the replica starts")
}

/// The names in the store's directory.
fn on_file(dir: &Path) -> BTreeSet<String> {
    fs::read_dir(dir)
        .expect("the directory")
        .map(|entry| entry.expect("an entry").file_name())
        .map(|name| name.to_string_lossy().into_owned())
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
/// call, and its next apply is the next number.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_enforce_on_a_spec_store_runs_and_a_start_runs_it_again() {
    let dir = tempfile::tempdir().expect("a directory");
    let mut replica = start("store-0", dir.path());
    page(&replica, "/readiness", None).await;
    let nothing = read(&replica, 0).await.expect_err("nothing stands");
    assert_eq!(nothing.code(), Code::NotFound, "{nothing:?}");

    // A write is the leader's: the replica answers once it leads its lease.
    let led = until(
        || apply(&replica, ApplyMode::DryRun, LEAST_LOAD),
        Result::is_ok,
    );
    let would = led.await.expect("the leader answers").expect("a verdict");
    assert!(would.accepted && !would.read_only, "{would:?}");
    assert_eq!(would.generation, 1);
    assert_eq!(on_file(dir.path()), BTreeSet::new());

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
    let all = BTreeSet::from([name(1), name(2), name(3)]);
    assert_eq!(on_file(dir.path()), all);
}

/// A start the store refuses exits with the refusal in its log: beside a
/// spec file, by the two flags; beside a lease another replica can take; on
/// a path that leads to no directory; and on a newest generation that does
/// not run, by its number and its field. Nothing is served.
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

    let (code, log) = refused(&[OsStr::new("--spec-store"), nowhere.as_os_str()]);
    assert_eq!(code, Some(1), "{log}");
    assert!(log.contains("the path leads to no directory"), "{log}");

    fs::write(store.join(name(1)), unweighed()).expect("a generation on file");
    let (code, log) = on_store(&[]);
    assert_eq!(code, Some(1), "{log}");
    assert!(
        log.contains("generation 1") && log.contains("profiles.default.score[0]"),
        "the generation and its field are named: {log}"
    );
}
