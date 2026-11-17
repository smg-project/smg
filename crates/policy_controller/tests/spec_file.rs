//! The policy spec's file source end to end (design 4.6): this package's
//! binary started on a spec file laid out as a mounted ConfigMap, on
//! ephemeral ports, asked for its document by the generated client on the
//! control listener. A start on a valid file answers its bytes under
//! generation 1, or under the number the file writes. The mount is then
//! updated as a ConfigMap's is, a link swapped to a new directory, and the
//! read call answers the next generation; a changed document under the
//! standing line is rejected, counted once, and the running generation keeps
//! answering. A start on a document the validator refuses exits, naming the
//! file and the field. A start with no spec file runs as it always did: no
//! spec call is served and no spec series rendered.
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

use std::{
    ffi::OsStr,
    fmt::Debug,
    fs,
    future::Future,
    os::unix::fs::symlink,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant},
};

use net::http_get;
use replicas::{Lease, Replica, ROOM};
use smg_policy_controller::{
    elector::ElectionSettings,
    proto::{
        policy_controller_client::PolicyControllerClient, GetPolicySpecRequest, PolicySpecDocument,
    },
};
use tempfile::TempDir;
use tokio::time;
use tonic::{metadata::AsciiMetadataValue, Code, Request, Response, Status};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// Between two asks of a wait.
const PACE: Duration = Duration::from_millis(50);

/// A document in the form of the design's example: no `generation` line.
const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";
/// Another one.
const RANDOM: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

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
/// listener taking the name a call states, on `file` when one is given.
fn start(id: &str, lease: &TempDir, file: Option<&Path>) -> Replica {
    let mut flags = vec![OsStr::new("--identity-as-told"), OsStr::new("control")];
    if let Some(file) = file {
        flags.extend([OsStr::new("--spec-file"), file.as_os_str()]);
    }
    Replica::start_with(
        id,
        Lease::File(&lease.path().join("lease")),
        ElectionSettings::default(),
        0,
        &flags,
    )
    .expect("the replica starts")
}

/// One `GetPolicySpec` call on the control listener, as `operator-a`.
async fn read(replica: &Replica, generation: u64) -> Result<PolicySpecDocument, Status> {
    let mut client = PolicyControllerClient::connect(format!("http://{}", replica.control))
        .await
        .map_err(|error| Status::unavailable(error.to_string()))?;
    let mut request = Request::new(GetPolicySpecRequest { generation });
    request.metadata_mut().insert(
        "x-smg-principal",
        AsciiMetadataValue::from_static("operator-a"),
    );
    client
        .get_policy_spec(request)
        .await
        .map(Response::into_inner)
}

/// What `probe` answers once `met` holds for it, asked every [`PACE`]; the
/// last answer where the room runs out first. A probe that does not answer
/// is left at the room's end too: the wait is no longer than its room.
async fn until<T: Debug, F: Future<Output = T>>(
    mut probe: impl FnMut() -> F,
    met: impl Fn(&T) -> bool,
) -> Result<T, String> {
    // The runtime's clock: a case that holds it passes the room at once.
    let started = time::Instant::now();
    let mut last = None;
    loop {
        let left = ROOM.saturating_sub(started.elapsed());
        if left.is_zero() {
            // The room ran out between two probes: the last one answered.
            return Err(format!("not met in {ROOM:?}; the last answer: {last:?}"));
        }
        let Ok(answer) = time::timeout(left, probe()).await else {
            return Err(format!("no answer in {ROOM:?}; the last one: {last:?}"));
        };
        if met(&answer) {
            return Ok(answer);
        }
        last = Some(answer);
        time::sleep(PACE).await;
    }
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
/// is not told as a probe with no answer. The probe takes a pace, as a call
/// does, and the clock is held.
#[tokio::test(start_paused = true)]
async fn a_probe_never_met_ends_its_wait_by_its_last_answer() {
    let probe = || async {
        time::sleep(PACE).await;
        7_u8
    };
    let waited = time::timeout(2 * ROOM, until(probe, |_| false))
        .await
        .expect("the wait ends inside twice its room");
    let told = waited.expect_err("no answer was met");
    assert!(told.starts_with("not met in"), "{told}");
    assert!(told.ends_with("Some(7)"), "{told}");
}

/// The standing document once the read call answers it as `generation`.
async fn answers(replica: &Replica, generation: u64) -> PolicySpecDocument {
    let asked = until(
        || read(replica, 0),
        |answer| matches!(answer, Ok(document) if document.generation == generation),
    );
    asked
        .await
        .expect("the read call answers the generation")
        .expect("an answer")
}

/// The admin listener's page at `path` once it answers 200 with a `line`.
async fn page(replica: &Replica, path: &str, line: Option<&str>) -> String {
    let asked = until(
        || http_get(replica.admin, path),
        |page| match (page, line) {
            (Ok((200, body)), Some(line)) => body.lines().any(|rendered| rendered == line),
            (Ok((200, _)), None) => true,
            _ => false,
        },
    );
    asked.await.expect("the page").expect("an answer").1
}

/// The binary on a mounted file answers the file's bytes under generation
/// 1, with their digest and format, and renders the generation; a file that
/// writes its line starts under that number. A generation that does not
/// stand is not found.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_started_on_a_spec_file_answers_its_document() {
    let lease = tempfile::tempdir().expect("a directory");
    let mount = Mount::new(LEAST_LOAD);
    let replica = start("spec-0", &lease, Some(&mount.file()));

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
    let replica = start("spec-1", &lease, Some(&numbered_mount.file()));
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
    let replica = start("spec-watch", &lease, Some(&mount.file()));
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

/// With no spec file a replica runs as it always did: ready, no spec call
/// served, no spec series rendered.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_replica_with_no_spec_file_runs_as_it_did() {
    let lease = tempfile::tempdir().expect("a directory");
    let replica = start("spec-none", &lease, None);
    page(&replica, "/readiness", None).await;

    let unserved = read(&replica, 0).await.expect_err("no spec call is served");
    assert_eq!(unserved.code(), Code::Unimplemented, "{unserved:?}");
    assert_eq!(
        unserved.message(),
        "GetPolicySpec is not served by a controller started with no spec file"
    );
    let metrics = page(&replica, "/metrics", None).await;
    assert!(!metrics.contains("smg_policy_spec"), "{metrics}");
}
