//! The policy spec's file source end to end (design 4.6): this package's
//! binary started on a spec file laid out as a mounted ConfigMap, on
//! ephemeral ports, asked for its document by the generated client on the
//! control listener. A start on a valid file answers its bytes under
//! generation 1, or under the number the file writes. The mount is then
//! updated as a ConfigMap's is, a link swapped to a new directory, and the
//! read call answers the next generation; a changed document under the
//! standing line is rejected, counted once, and the running generation keeps
//! answering. A start on a document the validator refuses exits, naming the
//! file and the field.
//!
//! The cases wait on an answer, never on a clock: the watch looks once a
//! second, and a wait that outlasts its room fails with the last answer it
//! saw.
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
    ffi::OsStr,
    fs,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Instant,
};

use replicas::{Lease, Replica, ROOM};
use smg_policy_controller::service::{elector::ElectionSettings, proto::ApplyMode};
use spec::{answers, apply, as_operator, control, page, read, until, LEAST_LOAD, PACE, RANDOM};
use tempfile::TempDir;
use tokio::time;
use tonic::Code;

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// A directory laid out as a mounted ConfigMap: `policy.yaml` is a link
/// through `..data` into a version's directory, and an update swaps `..data`
/// to a new directory in one rename, then removes the old one.
struct Mount {
    dir: TempDir,
    published: u32,
}

impl Mount {
    fn new(document: &str) -> Self {
        let dir = tempfile::tempdir().expect("a directory");
        symlink("..data/policy.yaml", dir.path().join("policy.yaml")).expect("the file's link");
        let mut mount = Self { dir, published: 0 };
        mount.publish(document);
        mount
    }

    /// The path a controller is given.
    fn file(&self) -> PathBuf {
        self.dir.path().join("policy.yaml")
    }

    /// One update: the document in a new version's directory, `..data`
    /// swapped to it, the version before it removed.
    fn publish(&mut self, document: &str) {
        let root = self.dir.path();
        let old = (self.published > 0).then(|| root.join(format!("..version_{}", self.published)));
        self.published += 1;
        let version = format!("..version_{}", self.published);
        fs::create_dir(root.join(&version)).expect("the version's directory");
        fs::write(root.join(&version).join("policy.yaml"), document).expect("the document");
        symlink(&version, root.join("..data_tmp")).expect("the staged link");
        fs::rename(root.join("..data_tmp"), root.join("..data")).expect("the swap");
        if let Some(old) = old {
            fs::remove_dir_all(old).expect("the version before it goes");
        }
    }
}

/// A replica that leads itself over a lease file of its own, its control
/// listener taking the name a call states, on the spec file `file`.
fn start(id: &str, lease: &TempDir, file: &Path) -> Replica {
    let flags = [
        OsStr::new("--identity-as-told"),
        OsStr::new("control"),
        OsStr::new("--spec-file"),
        file.as_os_str(),
    ];
    Replica::start_with(
        id,
        Lease::File(&lease.path().join("lease")),
        ElectionSettings::default(),
        0,
        &flags,
    )
    .expect("the replica starts")
}

/// A probe that never answers ends its wait at the room's end, by name. The
/// clock is held, so the room passes at once.
#[tokio::test(start_paused = true)]
async fn a_probe_that_never_answers_ends_its_wait_at_the_rooms_end() {
    let waited = time::timeout(2 * ROOM, until(std::future::pending::<u8>, |_| true))
        .await
        .expect("the wait ends inside twice its room");
    let told = waited.expect_err("no probe answered");
    assert!(told.starts_with("no answer in"), "{told}");
}

/// A probe that answers and is never met ends its wait at the room's end by
/// that name, with its last answer: a room that ran out between two probes
/// is not told as a probe with no answer. The probe answers at once, so the
/// room ends between two probes and nowhere else, and the clock is held.
#[tokio::test(start_paused = true)]
async fn a_probe_never_met_ends_its_wait_by_its_last_answer() {
    let probe = || std::future::ready(7_u8);
    let waited = time::timeout(2 * ROOM, until(probe, |_| false))
        .await
        .expect("the wait ends inside twice its room");
    let told = waited.expect_err("no answer was met");
    assert!(told.starts_with("not met in"), "{told}");
    assert!(told.ends_with("Some(7)"), "{told}");
}

/// The binary on a mounted file answers the file's bytes under generation
/// 1, with their digest and format, and renders the generation; a file that
/// writes its line starts under that number. A generation that does not
/// stand is not found.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_started_on_a_spec_file_answers_its_document() {
    let lease = tempfile::tempdir().expect("a directory");
    let mount = Mount::new(LEAST_LOAD);
    let replica = start("spec-0", &lease, &mount.file());

    let first = answers(&replica, 1).await;
    assert_eq!(first.document, LEAST_LOAD.as_bytes());
    assert_eq!(first.format, "yaml");
    assert!(first.digest.starts_with("sha256:"), "{}", first.digest);
    assert!(first.acks.is_empty());
    let other = read(&replica, 2).await.expect_err("no generation 2");
    assert_eq!(other.code(), Code::NotFound, "{other:?}");
    page(&replica, "/metrics", Some("smg_policy_spec_generation 1")).await;
    let applied = replica.logs(|replica| replica.logged("policy spec applied"));
    assert!(applied.await, "{}", replica.log());

    let lease = tempfile::tempdir().expect("a directory");
    let numbered = format!("generation: 41\n{LEAST_LOAD}");
    let numbered_mount = Mount::new(&numbered);
    let replica = start("spec-1", &lease, &numbered_mount.file());
    let written = answers(&replica, 41).await;
    assert_eq!(written.document, numbered.as_bytes());
    assert_eq!(
        written.digest, first.digest,
        "the line is not in the digest"
    );
}

/// The mount updated as a ConfigMap's is, the read call answers the next
/// generation with the new bytes, and the one before it is not retained. A
/// changed document under the standing line is rejected: one count, one
/// event, and the running generation keeps answering. A document under a
/// higher line runs under that line, and the series follow.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_swapped_link_under_a_running_replica_is_the_next_generation() {
    let lease = tempfile::tempdir().expect("a directory");
    let mut mount = Mount::new(LEAST_LOAD);
    let replica = start("spec-watch", &lease, &mount.file());
    let first = answers(&replica, 1).await;

    mount.publish(RANDOM);
    let second = answers(&replica, 2).await;
    assert_eq!(second.document, RANDOM.as_bytes());
    assert_ne!(second.digest, first.digest);
    let older = read(&replica, 1)
        .await
        .expect_err("generation 1 is not retained");
    assert_eq!(older.code(), Code::NotFound, "{older:?}");

    mount.publish(&format!("generation: 2\n{LEAST_LOAD}"));
    let rejected = r#"smg_policy_spec_apply_total{result="rejected"} 1"#;
    page(&replica, "/metrics", Some(rejected)).await;
    let running = read(&replica, 0).await.expect("the running generation");
    assert_eq!(
        (running.generation, running.document.as_slice()),
        (2, RANDOM.as_bytes())
    );
    let event = replica
        .logs(|replica| replica.logged("policy spec rejected: the running generation stays"));
    assert!(event.await, "{}", replica.log());

    mount.publish(&format!("generation: 7\n{LEAST_LOAD}"));
    let third = answers(&replica, 7).await;
    assert_eq!(third.digest, first.digest, "one document, one digest");
    let metrics = page(&replica, "/metrics", Some("smg_policy_spec_generation 7")).await;
    for line in [
        r#"smg_policy_spec_apply_total{result="applied"} 3"#,
        rejected,
    ] {
        assert!(
            metrics.lines().any(|rendered| rendered == line),
            "no line {line:?} in {metrics}"
        );
    }
}

/// A dry run on a running replica answers its document's verdict and writes
/// nothing: the next number for a document that would run, the field for
/// one that would not, and `read_only` for an enforce. The standing
/// generation keeps answering its document.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_dry_run_on_a_running_replica_answers_its_verdict_and_writes_nothing() {
    let lease = tempfile::tempdir().expect("a directory");
    let mount = Mount::new(LEAST_LOAD);
    let replica = start("spec-dry", &lease, &mount.file());
    let first = answers(&replica, 1).await;

    // A write is the leader's: the replica answers once it leads its lease.
    let led = until(|| apply(&replica, ApplyMode::DryRun, RANDOM), Result::is_ok);
    let would = led.await.expect("the leader answers").expect("a verdict");
    assert!(would.accepted && would.read_only, "{would:?}");
    assert_eq!(would.generation, 2);
    // The feed before and after: the standing document reads the loads, and
    // the offered one reads nothing.
    let feed = would.needs.expect("the needs diff");
    let before = feed.before.expect("the feed before");
    assert!(
        !before.models.is_empty() && before.models.iter().all(|row| row.loads),
        "{before:?}"
    );
    assert!(feed.after.expect("the feed after").models.is_empty());

    let unweighed = LEAST_LOAD.replace("[least-load]", "[{ least-load: { weight: -1.0 } }]");
    let refused = apply(&replica, ApplyMode::DryRun, &unweighed)
        .await
        .expect("a verdict");
    assert!(!refused.accepted, "{refused:?}");
    assert_eq!((refused.generation, &refused.digest), (1, &first.digest));
    let [issue] = refused.issues.as_slice() else {
        panic!("one issue: {refused:?}")
    };
    assert!(
        issue.field_path.starts_with("profiles.default.score[0]"),
        "{issue:?}"
    );

    let enforced = apply(&replica, ApplyMode::Enforce, RANDOM)
        .await
        .expect("an answer");
    assert!(enforced.read_only && !enforced.accepted, "{enforced:?}");

    let standing = read(&replica, 0).await.expect("the standing document");
    assert_eq!(
        (standing.generation, standing.document.as_slice()),
        (1, LEAST_LOAD.as_bytes())
    );
}

/// A start on a document the validator refuses exits with the refusal in
/// its log: the file, and the field by its path. Nothing is served.
#[test]
fn a_start_on_a_document_the_validator_refuses_exits_naming_the_field() {
    let mount =
        Mount::new(&LEAST_LOAD.replace("[least-load]", "[{ least-load: { weight: -1.0 } }]"));
    let file = mount.file();
    let mut child = Command::new(BIN)
        .args(["--controller-id", "spec-refused"])
        .args(["--runtime-worker-threads", "1"])
        .args(["--gateway-listen", "127.0.0.1:0"])
        .args(["--control-listen", "127.0.0.1:0"])
        .args(["--replication-listen", "127.0.0.1:0"])
        .args(["--admin-listen", "127.0.0.1:0"])
        .arg("--spec-file")
        .arg(&file)
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
            panic!("the start was not refused in {ROOM:?}");
        }
        thread::sleep(PACE);
    }
    let output = child.wait_with_output().expect("the child's output");
    let log = String::from_utf8_lossy(&[output.stdout, output.stderr].concat()).into_owned();
    assert_eq!(output.status.code(), Some(1), "{log}");
    assert!(log.contains("start refused"), "{log}");
    assert!(
        log.contains(&file.display().to_string()),
        "the file is named: {log}"
    );
    assert!(
        log.contains("profiles.default.score[0]"),
        "the field is named by its path: {log}"
    );
}

/// A running replica's scrape reads its store's counts: a worker whose
/// engine family the document's list does not name is counted at the scrape
/// after its registration, and no load write has moved a verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_on_a_listing_file_counts_the_member_its_list_does_not_name() {
    use openai_protocol::worker::RuntimeType;
    use smg_policy_controller::service::proto::{
        register_worker_request::Mode, HealthPolicy, RegisterWorkerRequest, Role, WireMode,
        WorkerSpec,
    };

    let lease = tempfile::tempdir().expect("a directory");
    let listing = format!("{LEAST_LOAD}view:\n  engines: [other]\n");
    let mount = Mount::new(&listing);
    let replica = start("spec-listing", &lease, &mount.file());
    let none = "smg_policy_controller_members_outside_list{list=\"engines\"} 0";
    let one = "smg_policy_controller_members_outside_list{list=\"engines\"} 1";
    page(&replica, "/metrics", Some(none)).await;

    // Its checks off, the worker is ready from its registration.
    let spec = WorkerSpec {
        url: "http://127.0.0.1:9/outside".to_owned(),
        wire: WireMode::Grpc as i32,
        role: Role::Single as i32,
        runtime: RuntimeType::Vllm.as_str().to_owned(),
        models: vec!["model-a".to_owned()],
        health_policy: Some(HealthPolicy {
            disabled: true,
            ..Default::default()
        }),
        ..Default::default()
    };
    let ask = RegisterWorkerRequest {
        spec: Some(spec),
        mode: Mode::Add as i32,
        wait_ready: false,
    };
    // A replica that does not lead yet refuses the registration and writes
    // nothing, so the ask is made until the leader takes it.
    let registered = until(
        || async {
            let mut operator = control(&replica).await?;
            operator.register_worker(as_operator(ask.clone())).await
        },
        Result::is_ok,
    );
    registered.await.expect("an answer").expect("registered");
    let scrape = page(&replica, "/metrics", Some(one)).await;
    let flips = "smg_policy_controller_overload_flips_total 0";
    assert!(scrape.lines().any(|line| line == flips), "{scrape}");
}
