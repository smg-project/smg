//! The worker object a row makes: the module's one table, as code.

use std::sync::Arc;

use smg_worker::{
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

/// The store's role for a worker object.
pub(super) fn role(worker_type: WorkerType) -> Role {
    Role::from(worker_type)
}

/// The address the row's object dials, as written. A row that names no
/// engine dials its key.
pub(super) fn dial_address(facts: &WorkerFacts) -> &str {
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
fn checks(policy: &HealthPolicy, drain_settle_secs: Option<u32>) -> HealthCheckConfig {
    let default = HealthCheckConfig::default();
    HealthCheckConfig {
        timeout_secs: seconds(policy.timeout_ms),
        check_interval_secs: seconds(policy.check_interval_ms),
        success_threshold: policy.success_threshold,
        failure_threshold: policy.failure_threshold,
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
mod tests {
    use std::{
        collections::{BTreeMap, HashMap},
        time::Instant,
    };

    use smg_worker::worker::RuntimeType;

    use super::*;
    use crate::{
        engine::view::threshold_permille,
        store::{identity, Registrant},
    };

    const URL: &str = "http://w:8000";

    fn row(facts: WorkerFacts) -> WorkerRow {
        WorkerRow {
            id: identity(&facts.key, facts.incarnation),
            facts,
            status: WorkerStatus::Pending,
            revision: 1,
            admitted_at: Instant::now(),
            maker: Registrant::Control,
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

    /// A row's own thresholds as an object states them read back as the row
    /// stated them: every permille from one to a thousand comes back from
    /// its ratio, and 800 permille is the 0.8 an engine reports, not a value
    /// beside it. A row that names none states none.
    #[test]
    fn a_rows_thresholds_read_back_from_the_ratio_an_object_states() {
        for permille in 1..=1_000_u16 {
            let stated = thresholds(&Overload {
                kv_usage_permille: Some(permille),
                waiting_requests: Some(u32::from(permille)),
            });
            assert_eq!(stated.waiting_requests, Some(usize::from(permille)));
            let usage = stated.token_usage.expect("a usage");
            assert_eq!(threshold_permille(usage), permille);
        }
        let stated = thresholds(&Overload {
            kv_usage_permille: Some(800),
            waiting_requests: None,
        });
        assert_eq!(stated.token_usage, Some(0.8));
        assert_eq!(thresholds(&Overload::default()), OverloadUpdate::default());
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

    /// A row under its engine's own address that states a width is the
    /// engine whole: the object holds the width and no rank.
    #[test]
    fn a_row_under_its_engines_address_holds_its_width_and_no_rank() {
        let made = object(&row(WorkerFacts {
            engine: URL.to_string(),
            dp_size: 4,
            ..at(URL)
        }));
        assert_eq!(made.metadata().spec.dp_base_url, None);
        assert_eq!((made.dp_rank(), made.dp_size()), (None, Some(4)));
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
