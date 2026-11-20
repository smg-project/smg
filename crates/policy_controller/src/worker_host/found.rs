//! The facts a worker object states of its row.

use smg_worker::{
    identity::basic::Worker,
    worker::{ConnectionMode, HealthCheckConfig},
};

use super::object::role;
use crate::{
    store::{HealthPolicy, Overload, WorkerFacts},
    view::{threshold_permille, WireModes},
};

/// The facts a provider states of a worker it found, from the object to the
/// row as the module's table describes. Facts the object does not carry
/// come from the standing row, or from the store's defaults. The models are
/// the ones the object serves now, and a threshold of its own registration
/// is stated over the standing row's.
pub fn facts(worker: &dyn Worker, standing: Option<&WorkerFacts>) -> WorkerFacts {
    let defaults = WorkerFacts::default();
    let standing = standing.unwrap_or(&defaults);
    let metadata = worker.metadata();
    let spec = &metadata.spec;
    let checks = &metadata.health_config;
    WorkerFacts {
        key: spec.url.clone(),
        engine: spec.dp_base_url.as_ref().unwrap_or(&spec.url).clone(),
        incarnation: standing.incarnation,
        role: role(spec.worker_type),
        wire_modes: match spec.connection_mode {
            ConnectionMode::Http => WireModes::HTTP,
            ConnectionMode::Grpc => WireModes::GRPC,
            ConnectionMode::Zmq => WireModes::ZMQ,
        },
        engine_kind: if spec.runtime_type.is_specified() {
            spec.runtime_type.as_str().to_owned()
        } else {
            String::new()
        },
        models: worker.models().into_iter().map(|card| card.id).collect(),
        // Values wider than a row's fields take the largest value they hold.
        dp_rank: u16::try_from(spec.dp_rank.unwrap_or(0)).unwrap_or(u16::MAX),
        dp_size: u16::try_from(spec.dp_size.unwrap_or(1)).unwrap_or(u16::MAX),
        labels: spec
            .labels
            .iter()
            .map(|(name, value)| (name.clone(), value.clone()))
            .collect(),
        zone: standing.zone.clone(),
        pairing: standing.pairing.clone(),
        encode: standing.encode.clone(),
        vision: standing.vision,
        max_running: standing.max_running,
        kv_block_size: spec
            .kv_block_size
            .map(|size| u32::try_from(size).unwrap_or(u32::MAX)),
        bootstrap_port: spec.bootstrap_port,
        priority: standing.priority,
        reachable_from: standing.reachable_from,
        signals: standing.signals,
        health: HealthPolicy {
            failure_threshold: checks.failure_threshold,
            success_threshold: checks.success_threshold,
            check_interval_ms: milliseconds(checks.check_interval_secs),
            timeout_ms: milliseconds(checks.timeout_secs),
            probes: !checks.disable_health_check,
        },
        overload: Overload {
            kv_usage_permille: spec
                .overload
                .token_usage
                .map(threshold_permille)
                .or(standing.overload.kv_usage_permille),
            waiting_requests: spec
                .overload
                .waiting_requests
                .map(|waiting| u32::try_from(waiting).unwrap_or(u32::MAX))
                .or(standing.overload.waiting_requests),
        },
        // A window the object's registration left at the default is not a
        // fact of the row: the object built from the row takes the default.
        drain_settle_secs: (checks.drain_settle_secs
            != HealthCheckConfig::default().drain_settle_secs)
            .then(|| u32::try_from(checks.drain_settle_secs).unwrap_or(u32::MAX)),
        demand_facts: standing.demand_facts,
    }
}

/// Convert the worker's seconds to the policy's milliseconds, up to the
/// largest interval the row can hold.
fn milliseconds(seconds: u64) -> u32 {
    u32::try_from(seconds.saturating_mul(1_000)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::{
        sync::Arc,
        time::{Duration, Instant},
    };

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        worker::{HealthCheckConfig, OverloadUpdate, WorkerSpec},
    };

    use super::facts;
    use crate::{
        demands::DemandFacts,
        store::{
            encoding::{EncodeHandoff, EncodeOffer},
            pairing::PairingDescriptor,
            Ask, GatewayId, HealthPolicy, LoadSink as _, LoadSource, Maker, MemoryStore, Overload,
            Reachability, Signals, ViewBuffers, WorkerFacts, WorkerRow, WorkerStatus,
            WorkersStore as _,
        },
        view::{LoadRow, Role, WireModes, WorkerId},
        worker_host::object,
    };

    const URL: &str = "http://worker:8000";

    fn row(facts: WorkerFacts) -> WorkerRow {
        WorkerRow {
            id: WorkerId(1),
            facts,
            status: WorkerStatus::Ready,
            revision: 1,
            admitted_at: Instant::now(),
            maker: Maker::Control,
        }
    }

    #[test]
    fn the_table_read_up_and_down_gives_the_object_back() {
        for (role, wire_modes) in [
            (Role::Single, WireModes::HTTP),
            (Role::Prefill, WireModes::GRPC),
            (Role::Decode, WireModes::GRPC),
            (Role::Encode, WireModes::ZMQ),
        ] {
            let facts_of_row = WorkerFacts {
                key: "http://worker:8000@1".to_owned(),
                engine: URL.to_owned(),
                engine_kind: "generic".to_owned(),
                role,
                wire_modes,
                models: vec!["a".to_owned(), "b".to_owned()],
                dp_rank: 1,
                dp_size: 2,
                labels: [("tier".to_owned(), "gold".to_owned())].into(),
                bootstrap_port: Some(8_998),
                kv_block_size: Some(32),
                drain_settle_secs: Some(30),
                health: HealthPolicy {
                    failure_threshold: 4,
                    success_threshold: 5,
                    check_interval_ms: 7_000,
                    timeout_ms: 2_000,
                    probes: false,
                },
                ..WorkerFacts::default()
            };
            let first = object(&row(facts_of_row.clone()));
            let found = facts(first.as_ref(), None);
            assert_eq!(found, facts_of_row);
            let again = object(&row(found));
            let (first, again) = (first.metadata(), again.metadata());
            assert_eq!(
                serde_json::to_value(&again.spec).expect("the spec"),
                serde_json::to_value(&first.spec).expect("the spec")
            );
            assert_eq!(
                serde_json::to_value(&again.health_config).expect("the checks"),
                serde_json::to_value(&first.health_config).expect("the checks")
            );
        }
    }

    #[test]
    fn a_fact_the_object_does_not_carry_is_the_standing_rows() {
        let spec = WorkerSpec {
            priority: 19,
            ..WorkerSpec::new(URL.to_owned())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec).build();
        let bare = facts(&worker, None);
        let expected = WorkerFacts {
            incarnation: 19,
            zone: Some("east".to_owned()),
            pairing: Some(Arc::new(PairingDescriptor {
                protocol: Some("pd".to_owned()),
                ..Default::default()
            })),
            encode: EncodeOffer {
                handoff: EncodeHandoff::Push,
                producer: true,
                ..EncodeOffer::default()
            },
            vision: true,
            max_running: Some(23),
            priority: -4,
            overload: Overload {
                kv_usage_permille: Some(800),
                waiting_requests: Some(12),
            },
            signals: Signals {
                events: true,
                loads: true,
                scrape_interval_ms: 3_000,
            },
            reachable_from: Reachability::Gateway(GatewayId(42)),
            demand_facts: DemandFacts { runtime: 7 },
            ..bare.clone()
        };
        let old = WorkerFacts {
            key: "http://old".to_owned(),
            role: Role::Decode,
            labels: [("old".to_owned(), "label".to_owned())].into(),
            ..expected.clone()
        };
        let defaults = WorkerFacts::default();
        assert_eq!(
            bare,
            WorkerFacts {
                key: URL.to_owned(),
                engine: URL.to_owned(),
                wire_modes: WireModes::HTTP,
                dp_size: 1,
                health: bare.health,
                ..defaults
            }
        );
        assert_eq!(facts(&worker, Some(&old)), expected);
    }

    /// A model list set after the object was built, and a threshold of its
    /// own registration: the row states both, and keeps the standing
    /// threshold the object leaves unset.
    #[test]
    fn an_object_states_the_models_it_serves_now_and_its_own_thresholds() {
        let spec = WorkerSpec {
            models: vec![ModelCard::new("first")].into(),
            overload: OverloadUpdate {
                waiting_requests: Some(7),
                token_usage: None,
            },
            ..WorkerSpec::new(URL.to_owned())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec).build();
        worker.set_models(vec![ModelCard::new("second")]);
        let standing = WorkerFacts {
            overload: Overload {
                kv_usage_permille: Some(800),
                waiting_requests: Some(12),
            },
            ..WorkerFacts::default()
        };
        let found = facts(&worker, Some(&standing));
        assert_eq!(found.models, ["second"]);
        let stated = Overload {
            kv_usage_permille: Some(800),
            waiting_requests: Some(7),
        };
        assert_eq!(found.overload, stated);
    }

    /// A threshold of the object's own under half a permille is held as
    /// one, as a document's is. At none the store reads every engine with
    /// a rank on file as over it, so an idle worker would be vetoed.
    #[test]
    fn a_threshold_under_half_a_permille_is_held_as_one_and_vetoes_no_idle_worker() {
        let spec = WorkerSpec {
            models: vec![ModelCard::new("model")].into(),
            overload: OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.0004),
            },
            ..WorkerSpec::new(URL.to_owned())
        };
        let found = facts(&<BasicWorkerBuilder>::from_spec(spec).build(), None);
        let stated = found.overload.kv_usage_permille;

        let store = MemoryStore::with_publish_window(Duration::ZERO);
        let now = Instant::now();
        let id = store.upsert(found).id();
        store.publish_now();
        store.record(id, 0, LoadRow::default(), now, LoadSource::Stream);
        store.publish_now();
        let ask = Ask {
            gateway: GatewayId(1),
            now,
            pinned: None,
            affinity: false,
            signal_cells: false,
        };
        let mut view = ViewBuffers::default();
        store
            .build_view("model", Role::Single, &ask, &mut view)
            .expect("the pool");
        assert!(!view.candidates[0].health.overloaded, "idle, and read over");
        assert_eq!(stated, Some(1));
    }

    #[test]
    fn a_wildcard_names_no_model_and_an_unset_width_is_one() {
        let worker = <BasicWorkerBuilder>::from_spec(WorkerSpec::new(URL.to_owned())).build();
        let found = facts(&worker, None);
        assert!(found.models.is_empty());
        assert!(found.engine_kind.is_empty());
        assert_eq!(
            (
                found.dp_rank,
                found.dp_size,
                found.kv_block_size,
                found.bootstrap_port
            ),
            (0, 1, None, None)
        );
        assert_eq!(found.engine, URL);
        assert_eq!(found.drain_settle_secs, None);

        let spec = WorkerSpec {
            dp_rank: Some(usize::from(u16::MAX) + 1),
            dp_size: Some(usize::from(u16::MAX) + 2),
            kv_block_size: Some(usize::MAX),
            ..WorkerSpec::new(URL.to_owned())
        };
        let worker = <BasicWorkerBuilder>::from_spec(spec).build();
        let found = facts(&worker, None);
        assert_eq!((found.dp_rank, found.dp_size), (u16::MAX, u16::MAX));
        assert_eq!(
            found.kv_block_size,
            Some(u32::try_from(usize::MAX).unwrap_or(u32::MAX))
        );
    }

    #[test]
    fn seconds_are_told_in_milliseconds() {
        let checks = HealthCheckConfig {
            check_interval_secs: 2,
            timeout_secs: 3,
            disable_health_check: true,
            success_threshold: 4,
            failure_threshold: 5,
            ..HealthCheckConfig::default()
        };
        let builder = || <BasicWorkerBuilder>::from_spec(WorkerSpec::new(URL.to_owned()));
        let worker = builder().health_config(checks.clone()).build();
        assert_eq!(
            facts(&worker, None).health,
            HealthPolicy {
                check_interval_ms: 2_000,
                timeout_ms: 3_000,
                success_threshold: 4,
                failure_threshold: 5,
                probes: false,
            }
        );
        let worker = builder()
            .health_config(HealthCheckConfig {
                check_interval_secs: u64::MAX,
                timeout_secs: u64::from(u32::MAX) / 1_000 + 1,
                disable_health_check: false,
                ..checks
            })
            .build();
        let health = facts(&worker, None).health;
        assert_eq!(
            (health.check_interval_ms, health.timeout_ms, health.probes),
            (u32::MAX, u32::MAX, true)
        );
    }
}
