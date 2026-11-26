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
//! changes the term is one ahead of, a restart under the same identity
//! counted as one. The spec stores whole seconds, so an election duration
//! with a fraction of a second is refused at the first write, and the wait
//! a release hands on is rounded up to them.
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

use super::{ConditionalWriteOutcome, ElectError, LeaseBackend, LeaseRecord};

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
/// closes by itself either way. The client's own retry of an answer that
/// says try later is the caller's `Config` too, and the client library turns
/// it on by default: a caller turns it off, or one call waits out the time
/// the answer names.
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

    fn create(&mut self, record: &LeaseRecord) -> Result<ConditionalWriteOutcome, ElectError> {
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
                Ok(ConditionalWriteOutcome::Done)
            }
            // Another candidate created the object first.
            Err(err) if status_code(&err) == Some(CONFLICT) => Ok(ConditionalWriteOutcome::Moved),
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
    ) -> Result<ConditionalWriteOutcome, ElectError> {
        let Some(expected) = expected else {
            return match self.seen {
                None => self.create(record),
                Some(_) => Ok(ConditionalWriteOutcome::Moved),
            };
        };
        for _ in 0..MOVES_UNDER_AN_UNCHANGED_RECORD {
            let Some(seen) = self.seen.clone() else {
                return Ok(ConditionalWriteOutcome::Moved);
            };
            if self.record_of(&seen)? != *expected {
                return Ok(ConditionalWriteOutcome::Moved);
            }
            match self.replace(&seen, record)? {
                Some(stored) => {
                    self.seen = Some(stored);
                    return Ok(ConditionalWriteOutcome::Done);
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
    // A holder's duration is a setting, refused with a fraction of a second.
    // A released record's is a wait handed on: it rounds up to the spec's
    // whole seconds, the safe side of a wait.
    let whole = if record.is_released() {
        record.duration_ms.div_ceil(1_000)
    } else if record.duration_ms.is_multiple_of(1_000) {
        record.duration_ms / 1_000
    } else {
        return Err(ElectError::Settings(
            "the duration must be whole seconds for a Kubernetes Lease",
        ));
    };
    let seconds = i32::try_from(whole)
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
#[path = "../../../tests/unit/service/elector/kubernetes/tests.rs"]
pub(in crate::service::elector) mod tests;
