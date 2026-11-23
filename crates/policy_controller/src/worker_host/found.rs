//! The facts a worker object states of its row.

use std::sync::Arc;

use smg_worker::{
    identity::basic::Worker,
    worker::{HealthCheckConfig, WorkerSpec},
};

use super::{membership_writer::Over, object::role};
use crate::{
    engine::view::{threshold_permille, WireModes},
    store::{
        pairing::{PairingDescriptor, LAYOUT_FACTS},
        HealthPolicy, Overload, WorkerFacts,
    },
};

/// The engine a worker's rows stand under: the address its ranks share, or
/// its own, as it was registered. A join base is not that key: it drops an
/// HTTP address's trailing slash.
pub(super) fn engine_of(worker: &dyn Worker) -> &str {
    let spec = &worker.metadata().spec;
    spec.dp_base_url.as_deref().unwrap_or(&spec.url)
}

/// The facts a provider states of a worker it found, from the object to the
/// row as the module's table describes. Facts the object does not carry
/// come from the standing row, or from the store's defaults. The models are
/// the ones the object serves now, and a threshold of its own registration
/// is stated over the standing row's. A threshold it leaves out stays the
/// standing row's: this reading keeps it under every verb. A door that
/// writes a replacement by id ([`Over::Replaced`]) clears it past this
/// reading.
///
/// The pairing descriptor is the standing row's here too. A host that
/// states the worker's own reads it of the worker ([`pairing_descriptor`]).
/// The controller's door states none: its log carries no descriptor.
pub fn facts(worker: &dyn Worker, standing: Option<&WorkerFacts>) -> WorkerFacts {
    let defaults = WorkerFacts::default();
    let standing = standing.unwrap_or(&defaults);
    let metadata = worker.metadata();
    let spec = &metadata.spec;
    let checks = &metadata.health_config;
    WorkerFacts {
        key: spec.url.clone(),
        engine: engine_of(worker).to_owned(),
        incarnation: standing.incarnation,
        role: role(spec.worker_type),
        wire_modes: WireModes::from(spec.connection_mode),
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
        overload: {
            let own = thresholds(spec);
            Overload {
                kv_usage_permille: own
                    .kv_usage_permille
                    .or(standing.overload.kv_usage_permille),
                waiting_requests: own.waiting_requests.or(standing.overload.waiting_requests),
            }
        },
        // A window the object's registration left at the default is not a
        // fact of the row: the object built from the row takes the default.
        drain_settle_secs: (checks.drain_settle_secs
            != HealthCheckConfig::default().drain_settle_secs)
            .then(|| u32::try_from(checks.drain_settle_secs).unwrap_or(u32::MAX)),
        demand_facts: standing.demand_facts,
    }
}

/// The thresholds a worker's own registration states.
fn thresholds(spec: &WorkerSpec) -> Overload {
    Overload {
        kv_usage_permille: spec.overload.token_usage.map(threshold_permille),
        waiting_requests: spec
            .overload
            .waiting_requests
            .map(|waiting| u32::try_from(waiting).unwrap_or(u32::MAX)),
    }
}

/// The facts `built` states at a write that names `over`: [`facts`], and at
/// a replacement by id the thresholds as its own registration states them.
/// A replacement is its maker's whole statement of what a registration can
/// state, so a threshold it leaves out is cleared. An add finds no row to
/// keep one from. An upsert and an update keep the standing row's.
#[cfg_attr(
    not(test),
    expect(dead_code, reason = "the replica's own door reads a worker so")
)]
pub(crate) fn stated(
    built: &dyn Worker,
    standing: Option<&WorkerFacts>,
    over: Option<Over<'_>>,
) -> WorkerFacts {
    let facts = facts(built, standing);
    match over {
        Some(Over::Replaced(_)) => WorkerFacts {
            overload: thresholds(&built.metadata().spec),
            ..facts
        },
        Some(Over::Updated(_)) | None => facts,
    }
}

/// The pairing facts `worker` offers a partner, as a row states them: the
/// protocol its operator set, its KV transport, its version, and its KV
/// layout facts in the store's order. The worker derived them once, at its
/// build, and a gateway's own pair index reads the same descriptor. So the
/// fields are mapped here and no label is read: one rule says what a worker
/// reports. A layout fact the store keeps no slot for is left out, since
/// its rule could not read it. `None` where the worker reports nothing, as
/// the rule reads a row with no descriptor.
///
/// A host that tells a worker it keeps no row of states them over
/// [`facts`]' reading of the worker alone.
pub fn pairing_descriptor(worker: &dyn Worker) -> Option<Arc<PairingDescriptor>> {
    let offered = worker.pd_pairing();
    let mut descriptor = PairingDescriptor {
        protocol: offered.explicit().map(str::to_owned),
        transport: offered.transport().map(str::to_owned),
        version: offered.version().map(str::to_owned),
        ..PairingDescriptor::default()
    };
    for (name, value) in offered.kv_layout() {
        let slot = LAYOUT_FACTS.iter().position(|&(_, fact)| fact == name);
        if let Some(slot) = slot {
            descriptor.kv_layout[slot] = Some(value.to_owned());
        }
    }
    descriptor.normalize();
    (descriptor != PairingDescriptor::default()).then(|| Arc::new(descriptor))
}

/// Convert the worker's seconds to the policy's milliseconds, up to the
/// largest interval the row can hold.
fn milliseconds(seconds: u64) -> u32 {
    u32::try_from(seconds.saturating_mul(1_000)).unwrap_or(u32::MAX)
}

#[cfg(test)]
mod tests {
    use std::{sync::Arc, time::Instant};

    use smg_worker::{
        identity::{basic::Worker, builder::BasicWorkerBuilder},
        model_card::ModelCard,
        pd::pairing::{layout_labels, PairingMismatch, PdPairingMode},
        worker::{HealthCheckConfig, OverloadUpdate, RuntimeType, WorkerSpec},
    };

    use super::{facts, pairing_descriptor};
    use crate::{
        document::spec::Pairing,
        engine::{
            demands::DemandFacts,
            view::{Role, WireModes, WorkerId},
        },
        store::{
            encoding::{EncodeHandoff, EncodeOffer},
            pairing::{mismatch, Member, PairingDescriptor, LAYOUT_FACTS},
            GatewayId, HealthPolicy, Overload, Reachability, Registrant, Signals, WorkerFacts,
            WorkerRow, WorkerStatus,
        },
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
            maker: Registrant::Control,
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
            demand_facts: DemandFacts {
                runtime: 7,
                ..DemandFacts::default()
            },
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

    /// A worker of `runtime` with `labels`.
    fn offering(runtime: RuntimeType, labels: &[(&str, &str)]) -> impl Worker {
        let mut spec = WorkerSpec::new(URL.to_owned());
        spec.runtime_type = runtime;
        spec.labels = labels
            .iter()
            .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
            .collect();
        <BasicWorkerBuilder>::from_spec(spec).build()
    }

    /// The descriptor is the one the worker derived at its build: each field
    /// in its place, a blank value dropped, and the four layout facts in the
    /// order the worker's own rule compares them. Each runtime's transport
    /// is its worker's. The table of every door keeps the standing row's.
    #[test]
    fn a_worker_offers_the_pairing_facts_it_derived() {
        let worker = offering(
            RuntimeType::Sglang,
            &[
                ("pairing_protocol", " pd "),
                ("disaggregation_transfer_backend", "NIXL"),
                ("version", "1.2"),
                ("kv_cache_dtype", "fp8"),
                ("block_size", "16"),
                ("attention_backend", "fa3"),
                ("model_dtype", "bf16"),
            ],
        );
        let offered = worker.pd_pairing();
        let stated = pairing_descriptor(&worker).expect("a descriptor");
        assert_eq!(stated.protocol.as_deref(), Some("pd"));
        assert_eq!(stated.transport.as_deref(), Some("nixl"));
        assert_eq!(stated.version.as_deref(), Some("1.2"));
        assert_eq!(stated.protocol.as_deref(), offered.explicit());
        assert_eq!(stated.transport.as_deref(), offered.transport());
        assert_eq!(stated.version.as_deref(), offered.version());
        let layout: Vec<_> = offered.kv_layout().map(|(_, value)| value).collect();
        assert_eq!(layout, ["fp8", "16", "fa3", "bf16"]);
        let slots = stated.kv_layout.iter().map(|value| value.as_deref());
        assert!(slots.eq(layout.into_iter().map(Some)));

        // Each runtime that names a transport states its worker's.
        for (runtime, labels, transport) in [
            (
                RuntimeType::Vllm,
                [("kv_connector", "NixlConnector")],
                "nixl",
            ),
            (
                RuntimeType::Sglang,
                [("disaggregation_transfer_backend", "nixl")],
                "nixl",
            ),
            (RuntimeType::TokenSpeed, [("tier", "gold")], "mooncake"),
        ] {
            let worker = offering(runtime, &labels);
            let stated = pairing_descriptor(&worker).expect("a descriptor");
            assert_eq!(stated.transport.as_deref(), Some(transport), "{runtime:?}");
            assert_eq!(stated.transport.as_deref(), worker.pd_pairing().transport());
        }

        // The row holds what the rule reads. A worker keeps a connector it
        // does not know as it stands, blanks and all, and the row reads the
        // name past them.
        let mut spec = WorkerSpec::new(URL.to_owned());
        spec.runtime_type = RuntimeType::Vllm;
        spec.kv_connector = Some(" Custom ".to_owned());
        let padded = <BasicWorkerBuilder>::from_spec(spec).build();
        assert_eq!(padded.pd_pairing().transport(), Some(" custom "));
        let read = pairing_descriptor(&padded).expect("a descriptor");
        assert_eq!(read.transport.as_deref(), Some("custom"));

        // A worker that reports nothing offers no descriptor.
        let silent = offering(RuntimeType::Sglang, &[("version", " ")]);
        assert_eq!(pairing_descriptor(&silent), None);
        let standing = WorkerFacts {
            pairing: Some(stated),
            ..WorkerFacts::default()
        };
        // The table states the standing row's descriptor, and none of its own.
        assert_eq!(facts(&worker, None).pairing, None);
        assert_eq!(facts(&silent, Some(&standing)).pairing, standing.pairing);
    }

    /// The store's layout facts are the worker crate's, name for name and in
    /// its order. The map leaves out a fact the store keeps no slot for, so
    /// a fact added there with no slot here fails this case.
    #[test]
    fn the_stores_layout_facts_are_the_workers_own() {
        let slots = LAYOUT_FACTS.iter().map(|&(_, fact)| fact);
        assert!(layout_labels().eq(slots));
    }

    /// The store's rule over the rows of two workers names the mismatch the
    /// workers' own rule names, in each mode: a row that states its worker's
    /// descriptor states every fact the rule reads.
    #[test]
    fn two_rows_pair_as_their_workers_do() {
        fn member(row: &WorkerFacts) -> Member<'_> {
            Member::new(
                WorkerId(1),
                row.wire_modes,
                &row.engine_kind,
                row.pairing.as_deref(),
            )
        }
        let labelled = |labels: &[(&str, &str)]| offering(RuntimeType::Sglang, labels);
        let fleet = [
            labelled(&[]),
            labelled(&[("pairing_protocol", "a")]),
            labelled(&[("pairing_protocol", "b")]),
            labelled(&[("disaggregation_transfer_backend", "nixl")]),
            labelled(&[("disaggregation_transfer_backend", "mooncake")]),
            labelled(&[("version", "1")]),
            labelled(&[("version", "2")]),
            labelled(&[("kv_cache_dtype", "fp8")]),
            labelled(&[("kv_cache_dtype", "auto")]),
            labelled(&[("page_size", "16")]),
            labelled(&[("block_size", "32")]),
            labelled(&[("attention_backend", "fa3")]),
            labelled(&[("model_dtype", "bf16")]),
            labelled(&[
                ("disaggregation_transfer_backend", "nixl"),
                ("version", "1"),
                ("page_size", "16"),
            ]),
            offering(RuntimeType::Vllm, &[("kv_connector", "NixlConnector")]),
            offering(RuntimeType::Vllm, &[]),
            offering(RuntimeType::TokenSpeed, &[]),
            offering(
                RuntimeType::TokenSpeed,
                &[("disaggregation_transfer_backend", "nixl")],
            ),
            offering(RuntimeType::Unspecified, &[("version", "1")]),
        ];
        let rows: Vec<_> = fleet
            .iter()
            .map(|worker| WorkerFacts {
                pairing: pairing_descriptor(worker),
                ..facts(worker, None)
            })
            .collect();
        let modes = [
            (Pairing::Strict, PdPairingMode::Strict),
            (Pairing::Lenient, PdPairingMode::Lenient),
            (Pairing::Off, PdPairingMode::Off),
        ];
        for (rule, classic) in modes {
            for (prefill, prefill_row) in fleet.iter().zip(&rows) {
                for (decode, decode_row) in fleet.iter().zip(&rows) {
                    let by_rows = mismatch(&member(prefill_row), &member(decode_row), rule)
                        .map(|component| component.to_string());
                    let by_workers = prefill
                        .pd_pairing()
                        .mismatch(decode.pd_pairing(), classic)
                        .map(PairingMismatch::describe);
                    assert_eq!(by_rows, by_workers, "{rule:?}");
                }
            }
        }
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
    /// one.
    #[test]
    fn a_threshold_under_half_a_permille_is_held_as_one() {
        let spec = WorkerSpec {
            models: vec![ModelCard::new("model")].into(),
            overload: OverloadUpdate {
                waiting_requests: None,
                token_usage: Some(0.0004),
            },
            ..WorkerSpec::new(URL.to_owned())
        };
        let found = facts(&<BasicWorkerBuilder>::from_spec(spec).build(), None);
        assert_eq!(found.overload.kv_usage_permille, Some(1));
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
