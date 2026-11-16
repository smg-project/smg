//! The policy spec's file source end to end (design 4.6): this package's
//! binary started on a spec file laid out as a mounted ConfigMap, on
//! ephemeral ports, asked for its document by the generated client on the
//! control listener. A start on a valid file answers its bytes under
//! generation 1, or under the number the file writes. A start on a document
//! the validator refuses exits, naming the file and the field. A start with
//! no spec file runs as it always did: no spec call is served and no spec
//! series rendered.
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
use tonic::{metadata::AsciiMetadataValue, Code, Request, Response, Status};

const BIN: &str = env!("CARGO_BIN_EXE_smg-policy-controller");

/// Between two asks of a wait.
const PACE: Duration = Duration::from_millis(50);

/// A document in the form of the design's example: no `generation` line.
const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";

/// A directory laid out as a mounted ConfigMap, and the path a controller
/// is given: `policy.yaml` is a link through `..data` into a version's
/// directory.
fn mounted(document: &str) -> (TempDir, PathBuf) {
    let dir = tempfile::tempdir().expect("a directory");
    let root = dir.path();
    fs::create_dir(root.join("..version_1")).expect("the version's directory");
    fs::write(root.join("..version_1/policy.yaml"), document).expect("the document");
    symlink("..version_1", root.join("..data")).expect("the data link");
    symlink("..data/policy.yaml", root.join("policy.yaml")).expect("the file's link");
    let file = root.join("policy.yaml");
    (dir, file)
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
/// last answer where the room runs out first.
async fn until<T: Debug, F: Future<Output = T>>(
    mut probe: impl FnMut() -> F,
    met: impl Fn(&T) -> bool,
) -> Result<T, String> {
    let started = Instant::now();
    loop {
        let answer = probe().await;
        if met(&answer) {
            return Ok(answer);
        }
        if started.elapsed() > ROOM {
            return Err(format!("not met in {ROOM:?}; the last answer: {answer:?}"));
        }
        tokio::time::sleep(PACE).await;
    }
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
    let (_mount, file) = mounted(LEAST_LOAD);
    let replica = start("spec-0", &lease, Some(&file));

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
    let (_mount, file) = mounted(&numbered);
    let replica = start("spec-1", &lease, Some(&file));
    let written = answers(&replica, 41).await;
    assert_eq!(written.document, numbered.as_bytes());
    assert_eq!(
        written.digest, first.digest,
        "the line is not in the digest"
    );
}

/// A start on a document the validator refuses exits with the refusal in
/// its log: the file, and the field by its path. Nothing is served.
#[test]
fn a_start_on_a_document_the_validator_refuses_exits_naming_the_field() {
    let (_mount, file) =
        mounted(&LEAST_LOAD.replace("[least-load]", "[{ least-load: { weight: -1.0 } }]"));
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
