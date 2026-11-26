use std::{
    convert::Infallible,
    sync::Arc,
    time::{Duration, Instant},
};

use http::{Method, Request, Response, StatusCode};
use kube::{client::Body, core::Status};
use parking_lot::Mutex;
use tokio::runtime::Runtime;

use super::*;
use crate::service::elector::{ElectionSettings, Elector, TickOutcome};

const S: Duration = Duration::from_secs(1);

/// How far ahead of the clock a case that steps through a wait starts its
/// instants: the elector dates a first sight at the later of the caller's
/// instant and its own clock, and an instant the real clock overtakes
/// moves the wait.
const AHEAD: Duration = S.saturating_mul(3_600);

/// One `Lease` on an API server in this process: the three calls the
/// backend makes, the resource version that makes a write conditional,
/// and the answers a real server gives.
struct Server {
    namespace: String,
    name: String,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    object: Option<Lease>,
    version: u64,
    /// Writes that land and then answer 503, while above zero.
    lost_answers: u32,
    /// Reads answered 403, while set.
    forbidden: bool,
    /// Every request's method and path, in order.
    calls: Vec<String>,
}

impl State {
    fn store(&mut self, mut object: Lease) -> Lease {
        self.version += 1;
        object.metadata.resource_version = Some(self.version.to_string());
        self.object = Some(object.clone());
        object
    }

    fn landed(&mut self, status: StatusCode, stored: &Lease) -> Response<Body> {
        if self.lost_answers > 0 {
            self.lost_answers -= 1;
            return failure(StatusCode::SERVICE_UNAVAILABLE, "ServiceUnavailable");
        }
        json(status, stored)
    }
}

impl Server {
    fn new(namespace: &str, name: &str) -> Arc<Self> {
        Arc::new(Self {
            namespace: namespace.to_string(),
            name: name.to_string(),
            state: Mutex::default(),
        })
    }

    /// A client whose every request this server answers.
    fn client(self: &Arc<Self>) -> Client {
        let server = Arc::clone(self);
        let service = tower::service_fn(move |request: Request<Body>| {
            let server = Arc::clone(&server);
            async move { Ok::<_, Infallible>(server.answer(request).await) }
        });
        Client::new(service, self.namespace.clone())
    }

    async fn answer(&self, request: Request<Body>) -> Response<Body> {
        let (parts, body) = request.into_parts();
        let bytes = body.collect_bytes().await.expect("the request body");
        let path = parts.uri.path().to_string();
        let collection = format!(
            "/apis/coordination.k8s.io/v1/namespaces/{}/leases",
            self.namespace
        );
        let item = format!("{collection}/{}", self.name);
        let mut state = self.state.lock();
        state.calls.push(format!("{} {path}", parts.method));
        match (parts.method, path) {
            (Method::GET, path) if path == item => {
                if state.forbidden {
                    return failure(StatusCode::FORBIDDEN, "Forbidden");
                }
                match &state.object {
                    Some(object) => json(StatusCode::OK, object),
                    None => failure(StatusCode::NOT_FOUND, "NotFound"),
                }
            }
            (Method::POST, path) if path == collection => {
                if state.object.is_some() {
                    return failure(StatusCode::CONFLICT, "AlreadyExists");
                }
                let object: Lease = serde_json::from_slice(&bytes).expect("a Lease");
                let stored = state.store(object);
                state.landed(StatusCode::CREATED, &stored)
            }
            (Method::PUT, path) if path == item => {
                let object: Lease = serde_json::from_slice(&bytes).expect("a Lease");
                let Some(current) = &state.object else {
                    return failure(StatusCode::NOT_FOUND, "NotFound");
                };
                if object.metadata.resource_version != current.metadata.resource_version {
                    return failure(StatusCode::CONFLICT, "Conflict");
                }
                let stored = state.store(object);
                state.landed(StatusCode::OK, &stored)
            }
            (method, path) => panic!("the backend sent {method} {path}"),
        }
    }

    /// An operator's label edit: the object's version moves, its record
    /// does not.
    fn touch(&self) {
        let mut state = self.state.lock();
        let mut object = state.object.take().expect("an object to touch");
        let label = state.version.to_string();
        object
            .metadata
            .labels
            .get_or_insert_with(BTreeMap::new)
            .insert("touched".to_string(), label);
        state.store(object);
    }

    /// An operator's deletion: the object is gone, whatever version a
    /// candidate read last.
    fn delete(&self) {
        self.state.lock().object = None;
    }

    fn object(&self) -> Lease {
        self.state.lock().object.clone().expect("an object")
    }

    fn calls(&self) -> Vec<String> {
        self.state.lock().calls.clone()
    }
}

fn json<T: serde::Serialize>(status: StatusCode, value: &T) -> Response<Body> {
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(serde_json::to_vec(value).expect("json")))
        .expect("a response")
}

fn failure(status: StatusCode, reason: &str) -> Response<Body> {
    let body = Status::failure(&format!("{reason}: the lease"), reason).with_code(status.as_u16());
    json(status, &body)
}

/// A runtime, a server and the backends over it. The runtime outlives
/// every backend declared after the cell.
pub(in crate::service::elector) struct Cell {
    runtime: Runtime,
    server: Arc<Server>,
}

impl Cell {
    pub(in crate::service::elector) fn new() -> Self {
        Self {
            runtime: Runtime::new().expect("a runtime"),
            server: Server::new("policy", "controller"),
        }
    }

    pub(in crate::service::elector) fn backend(&self) -> KubernetesLease {
        let server = Arc::clone(&self.server);
        // The client's worker is spawned on the runtime it is built in.
        let client = self.runtime.block_on(async move { server.client() });
        KubernetesLease::new(
            client,
            "policy",
            "controller",
            self.runtime.handle().clone(),
        )
    }

    fn candidate(&self, identity: &str) -> Elector {
        Elector::new(
            Box::new(self.backend()),
            identity,
            ElectionSettings::default(),
        )
        .expect("settings")
    }
}

fn record(holder: &str, term: u64, renewed_at_micros: u64) -> LeaseRecord {
    LeaseRecord {
        holder: holder.to_string(),
        term,
        acquired_at_micros: 1_700_000_000_000_001,
        renewed_at_micros,
        duration_ms: 3_000,
    }
}

#[test]
fn a_record_round_trips_through_the_lease_object() {
    let cell = Cell::new();
    let mut backend = cell.backend();
    assert_eq!(backend.read().expect("read"), None);
    let first = record("a@host:1", 1, 1_700_000_000_000_002);
    assert_eq!(
        backend.write(None, &first).expect("create"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(backend.read().expect("read"), Some(first.clone()));
    let object = cell.server.object();
    let spec = object.spec.clone().expect("a spec");
    assert_eq!(spec.holder_identity.as_deref(), Some("a@host:1"));
    assert_eq!(spec.lease_duration_seconds, Some(3));
    assert_eq!(spec.lease_transitions, Some(0));
    assert_eq!(
        spec.acquire_time
            .map(|MicroTime(stamp)| stamp.as_microsecond()),
        Some(1_700_000_000_000_001)
    );
    assert_eq!(
        spec.renew_time
            .map(|MicroTime(stamp)| stamp.as_microsecond()),
        Some(1_700_000_000_000_002)
    );
    assert_eq!(
        object
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(TERM_ANNOTATION))
            .map(String::as_str),
        Some("1")
    );
    // A release: no holder, no duration, the term kept.
    let released = LeaseRecord {
        holder: String::new(),
        duration_ms: 0,
        renewed_at_micros: 1_700_000_000_000_003,
        ..first.clone()
    };
    assert_eq!(
        backend.write(Some(&first), &released).expect("release"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(backend.read().expect("read"), Some(released));
    let spec = cell.server.object().spec.expect("a spec");
    assert_eq!(spec.holder_identity, None);
    assert_eq!(spec.lease_duration_seconds, None);
    assert!(cell
        .server
        .calls()
        .iter()
        .all(|call| call.contains("/namespaces/policy/leases")));
}

#[test]
fn an_object_without_a_record_is_a_released_lease() {
    let cell = Cell::new();
    // A chart created the object; nothing has held it.
    cell.server.state.lock().store(Lease {
        metadata: ObjectMeta {
            name: Some("controller".to_string()),
            namespace: Some("policy".to_string()),
            ..ObjectMeta::default()
        },
        spec: None,
    });
    let mut backend = cell.backend();
    let empty = backend.read().expect("read").expect("a record");
    assert!(empty.is_released());
    assert_eq!(empty.term, 0);
    let mut a = cell.candidate("a@host:1");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    // Term zero is no holder's word, and an apply that replaces the
    // object leaves this shape under a live leader: the fence stays
    // closed for one renew deadline, as over no record.
    assert_eq!(a.fence().leader_term(t0), None);
    let calls = cell.server.calls();
    assert!(calls.iter().any(|call| call.starts_with("PUT ")));
    assert!(!calls.iter().any(|call| call.starts_with("POST ")));
}

#[test]
fn a_write_is_conditional_on_the_version_read_last() {
    let cell = Cell::new();
    let mut a = cell.backend();
    let mut b = cell.backend();
    assert_eq!(a.read().expect("read"), None);
    assert_eq!(b.read().expect("read"), None);
    let first = record("a@host:1", 1, 10);
    assert_eq!(
        a.write(None, &first).expect("create"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        b.write(None, &record("b@host:2", 1, 11)).expect("create"),
        ConditionalWriteOutcome::Moved,
        "the object exists"
    );
    assert_eq!(b.read().expect("read"), Some(first.clone()));
    let second = record("a@host:1", 1, 12);
    assert_eq!(
        a.write(Some(&first), &second).expect("renew"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(
        b.write(Some(&first), &record("b@host:2", 2, 13))
            .expect("take"),
        ConditionalWriteOutcome::Moved,
        "the record moved under b's version"
    );
    assert_eq!(b.read().expect("read"), Some(second));
}

#[test]
fn an_object_deleted_between_the_read_and_the_write_is_a_move() {
    let cell = Cell::new();
    let mut a = cell.backend();
    let first = record("a@host:1", 1, 10);
    assert_eq!(
        a.write(None, &first).expect("create"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(a.read().expect("read"), Some(first.clone()));
    // An operator deletes the object under the holder: the renewal's
    // replace answers 404, the re-read finds nothing, and the write is
    // a move, never a recreation under the version read before it.
    cell.server.delete();
    assert_eq!(
        a.write(Some(&first), &record("a@host:1", 1, 11))
            .expect("renew"),
        ConditionalWriteOutcome::Moved
    );
    let methods: Vec<String> = cell
        .server
        .calls()
        .iter()
        .map(|call| call.split_whitespace().next().unwrap_or("").to_string())
        .collect();
    assert_eq!(methods, ["POST", "GET", "PUT", "GET"]);
    assert_eq!(a.read().expect("read"), None);
    assert_eq!(
        a.write(None, &record("a@host:1", 2, 12)).expect("take"),
        ConditionalWriteOutcome::Done,
        "the next take creates the object afresh"
    );
}

#[test]
fn a_metadata_edit_between_the_read_and_the_write_costs_the_holder_nothing() {
    let cell = Cell::new();
    let mut a = cell.backend();
    let first = record("a@host:1", 1, 10);
    assert_eq!(
        a.write(None, &first).expect("create"),
        ConditionalWriteOutcome::Done
    );
    assert_eq!(a.read().expect("read"), Some(first.clone()));
    cell.server.touch();
    let second = record("a@host:1", 1, 12);
    assert_eq!(
        a.write(Some(&first), &second).expect("renew"),
        ConditionalWriteOutcome::Done
    );
    let object = cell.server.object();
    assert_eq!(
        object
            .metadata
            .labels
            .as_ref()
            .and_then(|labels| labels.get("touched"))
            .map(String::as_str),
        Some("1"),
        "the edit survives the write"
    );
    assert_eq!(a.read().expect("read"), Some(second));
    let puts = cell
        .server
        .calls()
        .iter()
        .filter(|call| call.starts_with("PUT "))
        .count();
    assert_eq!(puts, 2, "one refused, one landed");
}

#[test]
fn a_write_that_landed_with_a_lost_answer_is_adopted_on_the_next_tick() {
    let cell = Cell::new();
    let mut a = cell.candidate("a@host:1");
    cell.server.state.lock().lost_answers = 1;
    let t0 = Instant::now() + AHEAD;
    let err = a.tick(t0).expect_err("the answer was lost");
    assert!(
        err.to_string()
            .starts_with("the lease at policy/controller: "),
        "{err}"
    );
    assert!(!a.is_holder());
    assert_eq!(
        cell.server
            .object()
            .spec
            .and_then(|spec| spec.holder_identity)
            .as_deref(),
        Some("a@host:1"),
        "the take landed"
    );
    assert_eq!(
        a.tick(t0 + S).expect("own"),
        TickOutcome::Acquired { term: 1 }
    );
    // No Lease record stood at the attempt, and an attempt in doubt
    // dates nothing: the fence opens one renew deadline after the write
    // that made the take this run's own. The take alone carries it
    // nowhere, and with one renewal behind it the fence opens by
    // itself, at no step.
    assert_eq!(a.fence().leader_term(t0 + 3 * S), None);
    assert_eq!(
        a.tick(t0 + 2 * S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert_eq!(a.fence().leader_term(t0 + 2 * S), None);
    assert_eq!(a.fence().leader_term(t0 + 3 * S), Some(1));
}

#[test]
fn a_duration_with_a_fraction_of_a_second_is_refused() {
    let cell = Cell::new();
    let settings = ElectionSettings {
        duration: Duration::from_millis(2_500),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_millis(1_500),
    };
    let mut a = Elector::new(Box::new(cell.backend()), "a@host:1", settings).expect("settings");
    let err = a.tick(Instant::now() + S).expect_err("whole seconds");
    assert_eq!(
        err.to_string(),
        "election settings: the duration must be whole seconds for a Kubernetes Lease"
    );
    assert!(!cell
        .server
        .calls()
        .iter()
        .any(|call| call.starts_with("POST ")));
}

#[test]
fn a_release_inside_the_takes_wait_is_written_in_whole_seconds() {
    // The wait such a release hands on is what its take still owed, and
    // the spec stores whole seconds: a wait with a fraction rounds up,
    // the safe side of a wait, where a holder's duration is refused.
    let cell = Cell::new();
    let settings = ElectionSettings {
        duration: Duration::from_secs(3),
        renew_period: Duration::from_millis(500),
        renew_deadline: Duration::from_millis(1_500),
    };
    let mut a = Elector::new(Box::new(cell.backend()), "a@host:1", settings).expect("settings");
    // A minute ahead of the clock: each write here crosses a socket.
    let t0 = Instant::now() + 60 * S;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    assert!(a.waits_to_lead());
    assert!(a.release().expect("release"));
    let spec = cell.server.object().spec.expect("a spec");
    assert_eq!(spec.holder_identity, None);
    // Half a second of the take's second and a half was still owed.
    assert_eq!(spec.lease_duration_seconds, Some(1));
}

#[test]
fn two_candidates_elect_over_one_lease_object() {
    let cell = Cell::new();
    let mut a = cell.candidate("a@host:1");
    let mut b = cell.candidate("b@host:2");
    let t0 = Instant::now() + AHEAD;
    assert_eq!(a.tick(t0).expect("take"), TickOutcome::Acquired { term: 1 });
    assert_eq!(
        b.tick(t0 + S / 2).expect("follow"),
        TickOutcome::Following {
            holder: "a@host:1".to_string(),
            term: 1
        }
    );
    assert_eq!(
        a.tick(t0 + S).expect("renew"),
        TickOutcome::Renewed { term: 1 }
    );
    // The holder stalls; b takes the lease once the record stood still
    // for its duration from b's first sight of the renewal.
    let first_sight = t0 + S + S / 2;
    assert!(matches!(
        b.tick(first_sight).expect("follow"),
        TickOutcome::Following { .. }
    ));
    assert!(matches!(
        b.tick(first_sight + 3 * S - Duration::from_micros(1))
            .expect("follow"),
        TickOutcome::Following { .. }
    ));
    assert_eq!(
        b.tick(first_sight + 3 * S).expect("take"),
        TickOutcome::Acquired { term: 2 }
    );
    assert_eq!(a.fence().leader_term(first_sight + 3 * S), None);
    assert_eq!(
        a.tick(first_sight + 4 * S).expect("look"),
        TickOutcome::Lost {
            term: 1,
            holder: "b@host:2".to_string()
        }
    );
    assert!(!a.release().expect("nothing to release"));
    let object = cell.server.object();
    let spec = object.spec.clone().expect("a spec");
    assert_eq!(spec.holder_identity.as_deref(), Some("b@host:2"));
    assert_eq!(spec.lease_transitions, Some(1));
    assert_eq!(
        object
            .metadata
            .annotations
            .and_then(|mut annotations| annotations.remove(TERM_ANNOTATION))
            .as_deref(),
        Some("2")
    );
    assert!(b.release().expect("release"));
    assert!(cell
        .server
        .object()
        .spec
        .and_then(|spec| spec.holder_identity)
        .is_none());
}

#[test]
fn a_term_annotation_this_build_cannot_read_is_malformed() {
    let cell = Cell::new();
    cell.server.state.lock().store(Lease {
        metadata: ObjectMeta {
            name: Some("controller".to_string()),
            namespace: Some("policy".to_string()),
            annotations: Some(BTreeMap::from([(
                TERM_ANNOTATION.to_string(),
                "seven".to_string(),
            )])),
            ..ObjectMeta::default()
        },
        spec: Some(LeaseSpec::default()),
    });
    let err = cell.backend().read().expect_err("malformed");
    assert!(matches!(&err, ElectError::Malformed { place, .. } if place == "policy/controller"));
    assert!(err.to_string().contains("smg.policy/term"), "{err}");
}

#[test]
fn a_server_refusal_names_the_place_and_the_reason() {
    let cell = Cell::new();
    cell.server.state.lock().forbidden = true;
    let err = cell.backend().read().expect_err("forbidden");
    let text = err.to_string();
    assert!(
        text.starts_with("the lease at policy/controller: "),
        "{text}"
    );
    assert!(text.contains("Forbidden"), "{text}");
}
