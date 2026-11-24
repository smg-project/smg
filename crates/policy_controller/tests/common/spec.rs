//! The policy spec's calls as a test binary makes them on a replica's
//! control listener, the two documents its cases hand over, and the waits
//! they share: on an answer, never on a clock, and no longer than the room.
//! A call's connection and its answer each end inside the room too.

use std::{fmt::Debug, future::Future, time::Duration};

use smg_policy_controller::service::proto::{
    policy_controller_client::PolicyControllerClient, ApplyMode, ApplyRequest, ApplyResult,
    GetPolicySpecRequest, PolicySpecDocument,
};
use tokio::time;
use tonic::{metadata::AsciiMetadataValue, Request, Response, Status};

use crate::{
    net::http_get,
    replicas::{Replica, ROOM},
};

/// Between two asks of a wait.
pub const PACE: Duration = Duration::from_millis(50);

/// A document in the form of the design's example: no `generation` line.
pub const LEAST_LOAD: &str = "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    score: [least-load]\n    pick: best\n";

/// Another one.
pub const RANDOM: &str =
    "apiVersion: smg.policy/v1\nprofiles:\n  default:\n    filter: [healthy]\n    pick: random\n";

/// `wait`'s answer inside the room: a wait that outlasts it fails the case,
/// by `what` it waited for.
pub async fn inside<T>(what: &str, wait: impl Future<Output = T>) -> T {
    time::timeout(ROOM, wait).await.expect(what)
}

/// A client of the replica's control listener.
pub async fn control(
    replica: &Replica,
) -> Result<PolicyControllerClient<tonic::transport::Channel>, Status> {
    let dial = PolicyControllerClient::connect(format!("http://{}", replica.control));
    inside("the control listener's connection", dial)
        .await
        .map_err(|error| Status::unavailable(error.to_string()))
}

/// `message` as a call `operator-a` makes.
pub fn as_operator<T>(message: T) -> Request<T> {
    let mut request = Request::new(message);
    request.metadata_mut().insert(
        "x-smg-principal",
        AsciiMetadataValue::from_static("operator-a"),
    );
    request
}

/// One `GetPolicySpec` call on the control listener, as `operator-a`.
pub async fn read(replica: &Replica, generation: u64) -> Result<PolicySpecDocument, Status> {
    let mut client = control(replica).await?;
    let call = client.get_policy_spec(as_operator(GetPolicySpecRequest { generation }));
    inside("the read call's answer", call)
        .await
        .map(Response::into_inner)
}

/// One `ApplyPolicySpec` call of a YAML document on the control listener,
/// as `operator-a`, under no guard.
pub async fn apply(
    replica: &Replica,
    mode: ApplyMode,
    document: &str,
) -> Result<ApplyResult, Status> {
    apply_over(replica, mode, document, "").await
}

/// [`apply`] under a guard: `expected` is the digest its caller read as
/// standing.
pub async fn apply_over(
    replica: &Replica,
    mode: ApplyMode,
    document: &str,
    expected: &str,
) -> Result<ApplyResult, Status> {
    let request = ApplyRequest {
        document: document.as_bytes().to_vec(),
        format: "yaml".to_owned(),
        mode: mode.into(),
        expected_digest: expected.to_owned(),
    };
    let mut client = control(replica).await?;
    let call = client.apply_policy_spec(as_operator(request));
    inside("the apply call's answer", call)
        .await
        .map(Response::into_inner)
}

/// What `probe` answers once `met` holds for it, asked every [`PACE`]; the
/// last answer where the room runs out first. A probe that does not answer
/// is left at the room's end too: the wait is no longer than its room.
pub async fn until<T: Debug, F: Future<Output = T>>(
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

/// The standing document once the read call answers it as `generation`.
pub async fn answers(replica: &Replica, generation: u64) -> PolicySpecDocument {
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
pub async fn page(replica: &Replica, path: &str, line: Option<&str>) -> String {
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
