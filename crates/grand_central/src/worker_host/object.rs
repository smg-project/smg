//! The worker object a row makes: the module's one table, as code.

use std::sync::Arc;

use smg_worker::{
    health::overload::OverloadThresholds,
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{
        ConnectionMode, HealthCheckConfig, OverloadUpdate, WorkerSpec,
        WorkerStatus as ObjectStatus, WorkerType,
    },
};

use crate::{
    engine::view::{Role, WireModes},
    store::{HealthPolicy, Overload, WorkerFacts, WorkerRow, WorkerStatus},
};

/// The worker object of `row`: every fact of the row a collector or the
/// connector reads, each in the field the module's table names, and the
/// row's status as the object's first. It is built with no connector and no
/// credential: neither is a row's fact.
///
/// The facts are taken apart with no rest pattern, so a fact added to the
/// row does not build until this function and the table name it.
pub fn object(row: &WorkerRow) -> Arc<dyn Worker> {
    Arc::new(builder(row).build())
}

/// [`object`] before its build. The holder can give it the host's connector
/// and credential.
pub(super) fn builder(row: &WorkerRow) -> BasicWorkerBuilder {
    let WorkerFacts {
        key,
        engine: _,
        incarnation: _,
        role,
        wire_modes,
        engine_kind,
        models,
        dp_rank,
        dp_size,
        labels,
        zone: _,
        pairing: _,
        encode: _,
        vision: _,
        max_running: _,
        kv_block_size,
        bootstrap_port,
        priority: _,
        reachable_from: _,
        signals: _,
        health,
        overload: _,
        drain_settle_secs,
        demand_facts: _,
    } = &row.facts;
    let mut spec = WorkerSpec::new(key.clone());
    spec.models = models.iter().map(ModelCard::new).collect::<Vec<_>>().into();
    spec.worker_type = match role {
        Role::Single => WorkerType::Regular,
        Role::Prefill => WorkerType::Prefill,
        Role::Decode => WorkerType::Decode,
        Role::Encode => WorkerType::Encode,
    };
    spec.connection_mode = wire(*wire_modes);
    spec.runtime_type = engine_kind.parse().unwrap_or_default();
    spec.labels = labels
        .iter()
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect();
    spec.kv_block_size = kv_block_size.and_then(|size| usize::try_from(size).ok());
    spec.bootstrap_port = *bootstrap_port;
    let address = dial_address(&row.facts);
    let ranked = address != key;
    if ranked {
        spec.dp_base_url = Some(address.to_owned());
    }
    if *dp_size > 1 {
        // A row under its engine's own address that states no rank is the
        // engine whole: the width it states is no rank's.
        let whole = !ranked && *dp_rank == 0;
        spec.dp_rank = (!whole).then(|| usize::from(*dp_rank));
        spec.dp_size = Some(usize::from(*dp_size));
    }
    <BasicWorkerBuilder>::from_spec(spec)
        .health_config(checks(health, *drain_settle_secs))
        .status(status(row.status))
}

/// [`builder`] with the checks off, whatever the row's policy says: what
/// the holder builds from for a row its host cannot dial.
pub(super) fn unchecked(row: &WorkerRow) -> BasicWorkerBuilder {
    let facts = &row.facts;
    builder(row).health_config(HealthCheckConfig {
        disable_health_check: true,
        ..checks(&facts.health, facts.drain_settle_secs)
    })
}

/// The row's own overload thresholds in the worker crate's form: what an
/// object states where its host's load monitor judges overload by it
/// ([`Objects::with_overload_thresholds`](super::Objects::with_overload_thresholds)).
/// The usage goes back from permille to the ratio an engine reports: the
/// object's own registration reads it back as the same permille
/// ([`facts`](super::facts)).
pub(super) fn thresholds(overload: &Overload) -> OverloadUpdate {
    OverloadUpdate {
        waiting_requests: overload
            .waiting_requests
            .map(|waiting| usize::try_from(waiting).unwrap_or(usize::MAX)),
        token_usage: overload
            .kv_usage_permille
            .map(|permille| f64::from(permille) / 1_000.0),
    }
}

/// Whether `facts` state the external runtime: an endpoint of another
/// party's API. A host's default thresholds are no such row's
/// ([`Objects::with_overload_thresholds`](super::Objects::with_overload_thresholds)).
pub(super) fn external(facts: &WorkerFacts) -> bool {
    facts.engine_kind.parse() == Ok(smg_worker::worker::RuntimeType::External)
}

/// Whether an object built from `facts` states a threshold of `defaults`:
/// an engine's row that names none for a signal `defaults` state. The rows
/// a host's overload switch changes
/// ([`Objects::state_overload_switch`](super::Objects::state_overload_switch)).
pub(super) fn takes_default(facts: &WorkerFacts, defaults: OverloadThresholds) -> bool {
    let Overload {
        kv_usage_permille,
        waiting_requests,
    } = facts.overload;
    let waiting = waiting_requests.is_none() && defaults.waiting_requests.is_some();
    let usage = kv_usage_permille.is_none() && defaults.token_usage.is_some();
    !external(facts) && (waiting || usage)
}

/// The store's role for a worker object.
pub(super) fn role(worker_type: WorkerType) -> Role {
    Role::from(worker_type)
}

/// The address the row's object dials, as written. A row that names no
/// engine dials its key.
pub(crate) fn dial_address(facts: &WorkerFacts) -> &str {
    if facts.engine.is_empty() {
        &facts.key
    } else {
        &facts.engine
    }
}

/// The one wire an object speaks, of the set its row holds: gRPC first, then
/// HTTP, then the host-local wire. A row that names none is dialed over
/// HTTP, the object's own default.
fn wire(modes: WireModes) -> ConnectionMode {
    if modes.contains(WireModes::GRPC) {
        ConnectionMode::Grpc
    } else if modes.contains(WireModes::ZMQ) && !modes.contains(WireModes::HTTP) {
        ConnectionMode::Zmq
    } else {
        ConnectionMode::Http
    }
}

/// The object's health checks under the row's policy, with the row's drain
/// window where it states one: else the window is the object's default.
/// A threshold is at least one: the object's state machine judges a row
/// that states none as one that states one, and no probe moves a worker by
/// a count of zero.
fn checks(policy: &HealthPolicy, drain_settle_secs: Option<u32>) -> HealthCheckConfig {
    let default = HealthCheckConfig::default();
    HealthCheckConfig {
        timeout_secs: seconds(policy.timeout_ms),
        check_interval_secs: seconds(policy.check_interval_ms),
        success_threshold: policy.success_threshold.max(1),
        failure_threshold: policy.failure_threshold.max(1),
        disable_health_check: !policy.probes,
        drain_settle_secs: drain_settle_secs.map_or(default.drain_settle_secs, u64::from),
    }
}

/// A policy's milliseconds in the whole seconds an object counts: a part of
/// a second is one, and so is none, since no check waits zero seconds.
fn seconds(milliseconds: u32) -> u64 {
    u64::from(milliseconds).div_ceil(1_000).max(1)
}

pub(super) fn status(status: WorkerStatus) -> ObjectStatus {
    match status {
        WorkerStatus::Pending => ObjectStatus::Pending,
        WorkerStatus::Ready => ObjectStatus::Ready,
        WorkerStatus::NotReady => ObjectStatus::NotReady,
        WorkerStatus::Failed => ObjectStatus::Failed,
        WorkerStatus::Draining => ObjectStatus::Draining,
    }
}

/// [`status`] read back: the store's status for an object's.
pub(super) fn row_status(status: ObjectStatus) -> WorkerStatus {
    match status {
        ObjectStatus::Pending => WorkerStatus::Pending,
        ObjectStatus::Ready => WorkerStatus::Ready,
        ObjectStatus::NotReady => WorkerStatus::NotReady,
        ObjectStatus::Failed => WorkerStatus::Failed,
        ObjectStatus::Draining => WorkerStatus::Draining,
    }
}

#[cfg(test)]
#[path = "../../tests/unit/worker_host/object/tests.rs"]
mod tests;
