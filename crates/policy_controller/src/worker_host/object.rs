//! The worker object a row makes: the module's one table, as code.

use std::sync::Arc;

use smg_worker::{
    identity::{basic::Worker, builder::BasicWorkerBuilder},
    model_card::ModelCard,
    worker::{
        ConnectionMode, HealthCheckConfig, WorkerSpec, WorkerStatus as ObjectStatus, WorkerType,
    },
};

use crate::{
    store::{HealthPolicy, WorkerFacts, WorkerRow, WorkerStatus},
    view::{Role, WireModes},
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

/// [`object`] before its build: what the holder builds from where its host
/// handed it a connector, which no row holds.
pub(super) fn builder(row: &WorkerRow) -> BasicWorkerBuilder {
    let WorkerFacts {
        key,
        engine,
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
    // The object dials its base address where it has one, else its URL: a
    // rank's row names its engine there, and a row that is its own engine
    // dials its key.
    if !engine.is_empty() && engine != key {
        spec.dp_base_url = Some(engine.clone());
    }
    if *dp_size > 1 {
        spec.dp_rank = Some(usize::from(*dp_rank));
        spec.dp_size = Some(usize::from(*dp_size));
    }
    <BasicWorkerBuilder>::from_spec(spec)
        .health_config(checks(health))
        .status(status(row.status))
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

/// The object's health checks under the row's policy. Every other setting of
/// the checks is the object's default.
fn checks(policy: &HealthPolicy) -> HealthCheckConfig {
    HealthCheckConfig {
        timeout_secs: seconds(policy.timeout_ms),
        check_interval_secs: seconds(policy.check_interval_ms),
        success_threshold: policy.success_threshold,
        failure_threshold: policy.failure_threshold,
        disable_health_check: !policy.probes,
        ..HealthCheckConfig::default()
    }
}

/// A policy's milliseconds in the whole seconds an object counts: a part of
/// a second is one, and so is none, since no check waits zero seconds.
fn seconds(milliseconds: u32) -> u64 {
    u64::from(milliseconds).div_ceil(1_000).max(1)
}

fn status(status: WorkerStatus) -> ObjectStatus {
    match status {
        WorkerStatus::Pending => ObjectStatus::Pending,
        WorkerStatus::Ready => ObjectStatus::Ready,
        WorkerStatus::NotReady => ObjectStatus::NotReady,
        WorkerStatus::Failed => ObjectStatus::Failed,
        WorkerStatus::Draining => ObjectStatus::Draining,
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{BTreeMap, HashMap},
        time::Instant,
    };

    use smg_worker::worker::RuntimeType;

    use super::*;
    use crate::store::{identity, Maker};

    const URL: &str = "http://w:8000";

    fn row(facts: WorkerFacts) -> WorkerRow {
        WorkerRow {
            id: identity(&facts.key, facts.incarnation),
            facts,
            status: WorkerStatus::Pending,
            revision: 1,
            admitted_at: Instant::now(),
            maker: Maker::Control,
        }
    }

    fn at(key: &str) -> WorkerFacts {
        WorkerFacts {
            key: key.to_string(),
            ..WorkerFacts::default()
        }
    }

    fn ids(object: &Arc<dyn Worker>) -> Vec<String> {
        object.models().into_iter().map(|card| card.id).collect()
    }

    /// The table, row by row: each fact the object carries reads back from
    /// the field the table names, and what no row states stays unsaid.
    #[test]
    fn every_fact_the_table_carries_reads_back_from_the_objects_field() {
        let facts = WorkerFacts {
            engine: "grpc://engine:9000".to_string(),
            incarnation: 3,
            role: Role::Prefill,
            wire_modes: WireModes::GRPC,
            engine_kind: "generic".to_string(),
            models: vec!["m".to_string(), "n".to_string()],
            dp_rank: 1,
            dp_size: 2,
            labels: BTreeMap::from([("tier".to_string(), "gold".to_string())]),
            kv_block_size: Some(32),
            bootstrap_port: Some(8_998),
            health: HealthPolicy {
                failure_threshold: 4,
                success_threshold: 5,
                check_interval_ms: 7_000,
                timeout_ms: 2_000,
                probes: true,
            },
            ..at("grpc://engine:9000@1")
        };
        let made = object(&WorkerRow {
            status: WorkerStatus::Ready,
            ..row(facts)
        });
        let spec = &made.metadata().spec;
        assert_eq!(made.url(), "grpc://engine:9000@1");
        assert_eq!(spec.dp_base_url.as_deref(), Some("grpc://engine:9000"));
        assert_eq!(made.base_url(), "grpc://engine:9000");
        assert_eq!(made.worker_type(), &WorkerType::Prefill);
        assert_eq!(made.connection_mode(), &ConnectionMode::Grpc);
        assert_eq!(spec.runtime_type, RuntimeType::Generic);
        assert_eq!(ids(&made), ["m", "n"]);
        assert_eq!((made.dp_rank(), made.dp_size()), (Some(1), Some(2)));
        let labels = HashMap::from([("tier".to_string(), "gold".to_string())]);
        assert_eq!(spec.labels, labels);
        assert_eq!(spec.kv_block_size, Some(32));
        assert_eq!(made.bootstrap_port(), Some(8_998));
        let checks = &made.metadata().health_config;
        assert_eq!((checks.failure_threshold, checks.success_threshold), (4, 5));
        assert_eq!((checks.check_interval_secs, checks.timeout_secs), (7, 2));
        assert!(!checks.disable_health_check);
        assert_eq!(made.status(), ObjectStatus::Ready);
        assert_eq!(made.api_key(), None);
        assert!(made.engine_connector().is_none());
    }

    /// A row that is its own engine, and one that names none, dials its key
    /// and holds no rank; what the row leaves unsaid the object does too.
    #[test]
    fn a_row_of_one_rank_dials_its_key_and_holds_no_rank() {
        for (engine, dp_size) in [(URL, 1), ("", 1), (URL, 0)] {
            let made = object(&row(WorkerFacts {
                engine: engine.to_string(),
                dp_size,
                ..at(URL)
            }));
            let spec = &made.metadata().spec;
            assert_eq!(spec.dp_base_url, None, "engine {engine:?}");
            assert_eq!(made.base_url(), URL);
            assert_eq!((made.dp_rank(), made.dp_size()), (None, None));
            assert_eq!((spec.kv_block_size, made.bootstrap_port()), (None, None));
            assert!(spec.labels.is_empty());
        }
    }

    #[test]
    fn an_object_speaks_one_wire_of_its_rows_set() {
        let (http, grpc, local) = (WireModes::HTTP, WireModes::GRPC, WireModes::ZMQ);
        for (modes, wire) in [
            (WireModes::NONE, ConnectionMode::Http),
            (http, ConnectionMode::Http),
            (grpc, ConnectionMode::Grpc),
            (local, ConnectionMode::Zmq),
            (http.union(grpc), ConnectionMode::Grpc),
            (http.union(local), ConnectionMode::Http),
            (grpc.union(local), ConnectionMode::Grpc),
            (http.union(grpc).union(local), ConnectionMode::Grpc),
        ] {
            let made = object(&row(WorkerFacts {
                wire_modes: modes,
                ..at(URL)
            }));
            assert_eq!(made.connection_mode(), &wire, "{modes:?}");
        }
    }

    #[test]
    fn a_rows_role_and_its_status_are_the_objects_own() {
        for (role, kind) in [
            (Role::Single, WorkerType::Regular),
            (Role::Prefill, WorkerType::Prefill),
            (Role::Decode, WorkerType::Decode),
            (Role::Encode, WorkerType::Encode),
        ] {
            let made = object(&row(WorkerFacts { role, ..at(URL) }));
            assert_eq!(made.worker_type(), &kind);
        }
        for (status, first) in [
            (WorkerStatus::Pending, ObjectStatus::Pending),
            (WorkerStatus::Ready, ObjectStatus::Ready),
            (WorkerStatus::NotReady, ObjectStatus::NotReady),
            (WorkerStatus::Failed, ObjectStatus::Failed),
            (WorkerStatus::Draining, ObjectStatus::Draining),
        ] {
            let made = object(&WorkerRow {
                status,
                ..row(at(URL))
            });
            assert_eq!(made.status(), first);
        }
    }

    /// And a row whose checks are off keeps the status its row holds: the
    /// object's own rule for one (ready at once) is not the store's.
    #[test]
    fn a_policy_in_milliseconds_is_checked_in_whole_seconds_and_never_none() {
        for (milliseconds, seconds) in [(0, 1), (1, 1), (1_000, 1), (1_001, 2), (60_000, 60)] {
            let made = object(&row(WorkerFacts {
                health: HealthPolicy {
                    check_interval_ms: milliseconds,
                    timeout_ms: milliseconds,
                    probes: false,
                    ..HealthPolicy::default()
                },
                ..at(URL)
            }));
            let checks = &made.metadata().health_config;
            assert_eq!(
                (checks.check_interval_secs, checks.timeout_secs),
                (seconds, seconds),
                "{milliseconds} ms"
            );
            assert!(checks.disable_health_check);
            assert_eq!(
                checks.drain_settle_secs,
                HealthCheckConfig::default().drain_settle_secs
            );
            assert_eq!(made.status(), ObjectStatus::Pending);
        }
    }

    #[test]
    fn a_row_of_no_model_is_a_wildcard_and_a_runtime_unknown_is_unspecified() {
        let made = object(&row(at(URL)));
        assert!(made.metadata().spec.models.is_wildcard());
        assert_eq!(made.metadata().spec.runtime_type, RuntimeType::Unspecified);
        let made = object(&row(WorkerFacts {
            models: vec!["m".to_string()],
            engine_kind: "no-such-runtime".to_string(),
            ..at(URL)
        }));
        assert_eq!(ids(&made), ["m"]);
        assert_eq!(made.metadata().spec.runtime_type, RuntimeType::Unspecified);
    }
}
