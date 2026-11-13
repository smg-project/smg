//! The Kubernetes backend of the elector, behind the `kubernetes` feature:
//! one `Lease` object of `coordination.k8s.io/v1` holds the record, and the
//! compare-and-swap is the object's resource version.
//!
//! The record maps onto the `Lease` spec field for field: the holder is
//! `holderIdentity`, the two wall timestamps are `acquireTime` and
//! `renewTime` (a `MicroTime`, exact to the microsecond), the duration is
//! `leaseDurationSeconds`, so the object reads to `kubectl` and to any other
//! leader election as a lease. The term has no spec field: it lives in the
//! [`TERM_ANNOTATION`], and `leaseTransitions` carries the count of holder
//! changes the term is one ahead of. The spec stores whole seconds, so an
//! election duration with a fraction of a second is refused at the first
//! write.
//!
//! A write replaces the object under the resource version of the object
//! read last; the server refuses a replace over a version that moved. An
//! object whose metadata alone moved (a label edit between the read and the
//! write) is read again and the write retried a bounded number of times, so
//! a holder loses nothing to an edit that left the record as it was.
//!
//! The trait is synchronous and the election thread is a plain thread: each
//! round trip blocks that thread on the runtime handle the backend was
//! given. A shard thread never waits on it.

use std::{collections::BTreeMap, fmt, io};

use k8s_openapi::{
    api::coordination::v1::{Lease, LeaseSpec},
    apimachinery::pkg::apis::meta::v1::{MicroTime, ObjectMeta},
    jiff::Timestamp,
};
use kube::{api::PostParams, Api, Client};
use tokio::runtime::Handle;

use super::{ElectError, LeaseBackend, LeaseRecord, Written};

/// The annotation on the `Lease` object that carries the fencing term.
pub const TERM_ANNOTATION: &str = "smg.policy/term";

/// How many times a write follows an object whose metadata moved under an
/// unchanged record before it gives up.
const MOVES_UNDER_AN_UNCHANGED_RECORD: usize = 3;

const CONFLICT: u16 = 409;
const NOT_FOUND: u16 = 404;

/// A lease record in a `Lease` object of `coordination.k8s.io/v1`.
///
/// The client is one the caller built inside its runtime and `handle` is
/// that runtime's: every call of the trait blocks the calling thread on it,
/// so the elector runs on a thread of its own, never on a runtime worker.
/// The runtime is a multi-thread one: `Handle::block_on` drives no I/O of a
/// current-thread runtime unless another thread runs it. The client's
/// timeouts are the caller's `Config`; a read timeout at or under the renew
/// deadline keeps a stalled round trip from outliving the fence, which
/// closes by itself either way.
pub struct KubernetesLease {
    api: Api<Lease>,
    namespace: String,
    name: String,
    handle: Handle,
    /// The object as the server returned it last, by a read or by this
    /// backend's own write: the resource version the next write is
    /// conditional on, and the metadata that write keeps.
    seen: Option<Lease>,
}

impl fmt::Debug for KubernetesLease {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("KubernetesLease")
            .field("namespace", &self.namespace)
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl KubernetesLease {
    /// The `Lease` named `name` in `namespace`, through `client`, with the
    /// runtime `handle` the election thread blocks on.
    pub fn new(client: Client, namespace: &str, name: &str, handle: Handle) -> Self {
        Self {
            api: Api::namespaced(client, namespace),
            namespace: namespace.to_string(),
            name: name.to_string(),
            handle,
            seen: None,
        }
    }

    /// The object's namespace.
    pub fn namespace(&self) -> &str {
        &self.namespace
    }

    /// The object's name.
    pub fn name(&self) -> &str {
        &self.name
    }

    fn fetch(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        let object = self
            .handle
            .block_on(self.api.get_opt(&self.name))
            .map_err(|err| self.io(err))?;
        let record = object
            .as_ref()
            .map(|object| self.record_of(object))
            .transpose()?;
        self.seen = object;
        Ok(record)
    }

    fn create(&mut self, record: &LeaseRecord) -> Result<Written, ElectError> {
        let object = Lease {
            metadata: ObjectMeta {
                name: Some(self.name.clone()),
                namespace: Some(self.namespace.clone()),
                annotations: Some(BTreeMap::from([(
                    TERM_ANNOTATION.to_string(),
                    record.term.to_string(),
                )])),
                ..ObjectMeta::default()
            },
            spec: Some(spec_of(record, LeaseSpec::default())?),
        };
        match self
            .handle
            .block_on(self.api.create(&PostParams::default(), &object))
        {
            Ok(stored) => {
                self.seen = Some(stored);
                Ok(Written::Done)
            }
            // Another candidate created the object first.
            Err(err) if status_code(&err) == Some(CONFLICT) => Ok(Written::Moved),
            Err(err) => Err(self.io(err)),
        }
    }

    /// Replace `seen` by `record` under its resource version; `None` when
    /// the object moved (its version, or the object itself) since `seen`.
    fn replace(&self, seen: &Lease, record: &LeaseRecord) -> Result<Option<Lease>, ElectError> {
        let mut object = seen.clone();
        object
            .metadata
            .annotations
            .get_or_insert_with(BTreeMap::new)
            .insert(TERM_ANNOTATION.to_string(), record.term.to_string());
        object.spec = Some(spec_of(record, seen.spec.clone().unwrap_or_default())?);
        match self.handle.block_on(
            self.api
                .replace(&self.name, &PostParams::default(), &object),
        ) {
            Ok(stored) => Ok(Some(stored)),
            Err(err) if matches!(status_code(&err), Some(CONFLICT | NOT_FOUND)) => Ok(None),
            Err(err) => Err(self.io(err)),
        }
    }

    fn record_of(&self, object: &Lease) -> Result<LeaseRecord, ElectError> {
        let spec = object.spec.as_ref();
        let term = match object
            .metadata
            .annotations
            .as_ref()
            .and_then(|annotations| annotations.get(TERM_ANNOTATION))
        {
            None => 0,
            Some(text) => text.parse().map_err(|err| {
                self.malformed(format!("the {TERM_ANNOTATION} annotation {text:?}: {err}"))
            })?,
        };
        let seconds = spec
            .and_then(|spec| spec.lease_duration_seconds)
            .unwrap_or(0);
        let duration_ms = u64::try_from(seconds)
            .map_err(|_| self.malformed(format!("leaseDurationSeconds {seconds}")))?
            .saturating_mul(1_000);
        Ok(LeaseRecord {
            holder: spec
                .and_then(|spec| spec.holder_identity.clone())
                .unwrap_or_default(),
            term,
            acquired_at_micros: micros_of(spec.and_then(|spec| spec.acquire_time.as_ref()))
                .map_err(|detail| self.malformed(detail))?,
            renewed_at_micros: micros_of(spec.and_then(|spec| spec.renew_time.as_ref()))
                .map_err(|detail| self.malformed(detail))?,
            duration_ms,
        })
    }

    fn io(&self, err: kube::Error) -> ElectError {
        ElectError::Io {
            place: self.place(),
            source: io::Error::other(err),
        }
    }

    fn malformed(&self, detail: String) -> ElectError {
        ElectError::Malformed {
            place: self.place(),
            detail,
        }
    }
}

impl LeaseBackend for KubernetesLease {
    fn read(&mut self) -> Result<Option<LeaseRecord>, ElectError> {
        self.fetch()
    }

    fn write(
        &mut self,
        expected: Option<&LeaseRecord>,
        record: &LeaseRecord,
    ) -> Result<Written, ElectError> {
        let Some(expected) = expected else {
            return match self.seen {
                None => self.create(record),
                Some(_) => Ok(Written::Moved),
            };
        };
        for _ in 0..MOVES_UNDER_AN_UNCHANGED_RECORD {
            let Some(seen) = self.seen.clone() else {
                return Ok(Written::Moved);
            };
            if self.record_of(&seen)? != *expected {
                return Ok(Written::Moved);
            }
            match self.replace(&seen, record)? {
                Some(stored) => {
                    self.seen = Some(stored);
                    return Ok(Written::Done);
                }
                // The object moved: by its record, which the next round
                // answers `Moved` for, or by its metadata alone, which the
                // next round writes over.
                None => {
                    self.fetch()?;
                }
            }
        }
        Err(ElectError::Io {
            place: self.place(),
            source: io::Error::other(
                "the Lease object's metadata keeps moving under an unchanged record",
            ),
        })
    }

    fn place(&self) -> String {
        format!("{}/{}", self.namespace, self.name)
    }
}

/// The spec for `record`, over `base` for the fields the record has no say
/// in (a preferred holder, a strategy).
fn spec_of(record: &LeaseRecord, base: LeaseSpec) -> Result<LeaseSpec, ElectError> {
    if !record.duration_ms.is_multiple_of(1_000) {
        return Err(ElectError::Settings(
            "the duration must be whole seconds for a Kubernetes Lease",
        ));
    }
    let seconds = i32::try_from(record.duration_ms / 1_000)
        .map_err(|_| ElectError::Settings("the duration must fit the Lease's seconds"))?;
    Ok(LeaseSpec {
        holder_identity: (!record.holder.is_empty()).then(|| record.holder.clone()),
        lease_duration_seconds: (seconds != 0).then_some(seconds),
        acquire_time: micro_time(record.acquired_at_micros).map_err(ElectError::Settings)?,
        renew_time: micro_time(record.renewed_at_micros).map_err(ElectError::Settings)?,
        lease_transitions: Some(i32::try_from(record.term.saturating_sub(1)).unwrap_or(i32::MAX)),
        ..base
    })
}

/// Microseconds since the Unix epoch of a `MicroTime`; `0` for none.
fn micros_of(time: Option<&MicroTime>) -> Result<u64, String> {
    match time {
        None => Ok(0),
        Some(MicroTime(stamp)) => u64::try_from(stamp.as_microsecond())
            .map_err(|_| format!("the timestamp {stamp} is before the epoch")),
    }
}

/// The `MicroTime` of microseconds since the Unix epoch; none for `0`.
fn micro_time(micros: u64) -> Result<Option<MicroTime>, &'static str> {
    if micros == 0 {
        return Ok(None);
    }
    i64::try_from(micros)
        .ok()
        .and_then(|micros| Timestamp::from_microsecond(micros).ok())
        .map(|stamp| Some(MicroTime(stamp)))
        .ok_or("the record's timestamp is past the Lease's range")
}

fn status_code(err: &kube::Error) -> Option<u16> {
    match err {
        kube::Error::Api(status) => Some(status.code),
        _ => None,
    }
}

#[cfg(test)]
pub(in crate::elector) mod tests {
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
    use crate::elector::{ElectionSettings, Elector, Tick};

    const S: Duration = Duration::from_secs(1);

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
        let body =
            Status::failure(&format!("{reason}: the lease"), reason).with_code(status.as_u16());
        json(status, &body)
    }

    /// A runtime, a server and the backends over it. The runtime outlives
    /// every backend declared after the cell.
    pub(in crate::elector) struct Cell {
        runtime: Runtime,
        server: Arc<Server>,
    }

    impl Cell {
        pub(in crate::elector) fn new() -> Self {
            Self {
                runtime: Runtime::new().expect("a runtime"),
                server: Server::new("policy", "controller"),
            }
        }

        pub(in crate::elector) fn backend(&self) -> KubernetesLease {
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
        assert_eq!(backend.write(None, &first).expect("create"), Written::Done);
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
            Written::Done
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
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("take"), Tick::Acquired { term: 1 });
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
        assert_eq!(a.write(None, &first).expect("create"), Written::Done);
        assert_eq!(
            b.write(None, &record("b@host:2", 1, 11)).expect("create"),
            Written::Moved,
            "the object exists"
        );
        assert_eq!(b.read().expect("read"), Some(first.clone()));
        let second = record("a@host:1", 1, 12);
        assert_eq!(
            a.write(Some(&first), &second).expect("renew"),
            Written::Done
        );
        assert_eq!(
            b.write(Some(&first), &record("b@host:2", 2, 13))
                .expect("take"),
            Written::Moved,
            "the record moved under b's version"
        );
        assert_eq!(b.read().expect("read"), Some(second));
    }

    #[test]
    fn an_object_deleted_between_the_read_and_the_write_is_a_move() {
        let cell = Cell::new();
        let mut a = cell.backend();
        let first = record("a@host:1", 1, 10);
        assert_eq!(a.write(None, &first).expect("create"), Written::Done);
        assert_eq!(a.read().expect("read"), Some(first.clone()));
        // An operator deletes the object under the holder: the renewal's
        // replace answers 404, the re-read finds nothing, and the write is
        // a move, never a recreation under the version read before it.
        cell.server.delete();
        assert_eq!(
            a.write(Some(&first), &record("a@host:1", 1, 11))
                .expect("renew"),
            Written::Moved
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
            Written::Done,
            "the next take creates the object afresh"
        );
    }

    #[test]
    fn a_metadata_edit_between_the_read_and_the_write_costs_the_holder_nothing() {
        let cell = Cell::new();
        let mut a = cell.backend();
        let first = record("a@host:1", 1, 10);
        assert_eq!(a.write(None, &first).expect("create"), Written::Done);
        assert_eq!(a.read().expect("read"), Some(first.clone()));
        cell.server.touch();
        let second = record("a@host:1", 1, 12);
        assert_eq!(
            a.write(Some(&first), &second).expect("renew"),
            Written::Done
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
        let t0 = Instant::now() + S;
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
        assert_eq!(a.tick(t0 + S).expect("own"), Tick::Acquired { term: 1 });
        assert_eq!(a.fence().leader_term(t0 + S), Some(1));
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
    fn two_candidates_elect_over_one_lease_object() {
        let cell = Cell::new();
        let mut a = cell.candidate("a@host:1");
        let mut b = cell.candidate("b@host:2");
        let t0 = Instant::now() + S;
        assert_eq!(a.tick(t0).expect("take"), Tick::Acquired { term: 1 });
        assert_eq!(
            b.tick(t0 + S / 2).expect("follow"),
            Tick::Following {
                holder: "a@host:1".to_string(),
                term: 1
            }
        );
        assert_eq!(a.tick(t0 + S).expect("renew"), Tick::Renewed { term: 1 });
        // The holder stalls; b takes the lease once the record stood still
        // for its duration from b's first sight of the renewal.
        let first_sight = t0 + S + S / 2;
        assert!(matches!(
            b.tick(first_sight).expect("follow"),
            Tick::Following { .. }
        ));
        assert!(matches!(
            b.tick(first_sight + 3 * S - Duration::from_micros(1))
                .expect("follow"),
            Tick::Following { .. }
        ));
        assert_eq!(
            b.tick(first_sight + 3 * S).expect("take"),
            Tick::Acquired { term: 2 }
        );
        assert_eq!(a.fence().leader_term(first_sight + 3 * S), None);
        assert_eq!(
            a.tick(first_sight + 4 * S).expect("look"),
            Tick::Lost {
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
        assert!(
            matches!(&err, ElectError::Malformed { place, .. } if place == "policy/controller")
        );
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
}
