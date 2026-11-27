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

/// A threshold a row states as none is one on the object: its state
/// machine moves a worker at the first outcome there, and never by a
/// count of zero.
#[test]
fn a_threshold_of_none_is_one_on_the_object() {
    let policy = HealthPolicy {
        failure_threshold: 0,
        success_threshold: 0,
        ..HealthPolicy::default()
    };
    let checks = checks(&policy, None);
    assert_eq!((checks.failure_threshold, checks.success_threshold), (1, 1));
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
