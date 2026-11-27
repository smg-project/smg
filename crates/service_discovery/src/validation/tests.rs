use std::collections::HashMap;

use super::*;

fn assert_invalid(config: impl Into<DiscoveryConfig>, field: &str, value: &str, reason: &str) {
    let result = config.into().validate(Roles::Regular);
    assert_eq!(
        result.map_err(|error| error.to_string()),
        Err(format!(
            "Invalid value for field '{field}': {value} - {reason}"
        ))
    );
}

fn assert_refused(config: impl Into<DiscoveryConfig>, roles: Roles, expected: &str) {
    let result = config.into().validate(roles);
    assert!(
        matches!(&result, Err(DiscoveryConfigError::ValidationFailed { reason }) if reason == expected),
        "{result:?}"
    );
}

fn selector() -> HashMap<String, String> {
    [("app".into(), "worker".into())].into()
}

fn kubernetes() -> KubernetesDiscoveryConfig {
    KubernetesDiscoveryConfig {
        selector: selector(),
        ..Default::default()
    }
}

fn file() -> FileDiscoveryConfig {
    FileDiscoveryConfig {
        path: "workers.json".into(),
        check_interval_secs: 30,
    }
}

fn slurm(names: [&[&str]; 4]) -> SlurmDiscoveryConfig {
    let [job_names, encode_job_names, prefill_job_names, decode_job_names] =
        names.map(|names| names.iter().map(|name| name.to_string()).collect());
    SlurmDiscoveryConfig {
        job_names,
        encode_job_names,
        prefill_job_names,
        decode_job_names,
        port: 8000,
        bootstrap_port: None,
        user: None,
        account: None,
        partition: None,
        check_interval_secs: 30,
    }
}

fn regular_slurm() -> SlurmDiscoveryConfig {
    slurm([&["worker"], &[], &[], &[]])
}

fn consul() -> ConsulDiscoveryConfig {
    ConsulDiscoveryConfig {
        service: "workers".into(),
        address: None,
        datacenter: None,
        check_interval_secs: 30,
    }
}

#[test]
fn kubernetes_requires_a_regular_selector_for_regular_roles() {
    assert_refused(
        KubernetesDiscoveryConfig::default(),
        Roles::Regular,
        "Regular mode with service discovery requires a non-empty selector",
    );
    assert!(DiscoveryConfig::from(kubernetes())
        .validate(Roles::Regular)
        .is_ok());
}

#[test]
fn kubernetes_requires_either_selector_for_the_pair() {
    let roles = Roles::PrefillDecode;
    let reason = "PD mode with service discovery requires at least one non-empty selector (prefill or decode)";
    assert_refused(kubernetes(), roles, reason);
    for (prefill_selector, decode_selector) in
        [(selector(), HashMap::new()), (HashMap::new(), selector())]
    {
        let config = KubernetesDiscoveryConfig {
            prefill_selector,
            decode_selector,
            ..Default::default()
        };
        assert!(DiscoveryConfig::from(config).validate(roles).is_ok());
    }
}

#[test]
fn kubernetes_requires_a_selector_for_each_of_the_three_roles() {
    let roles = Roles::EncodePrefillDecode;
    let reason = "EPD mode with service discovery requires non-empty encode_selector, prefill_selector, and decode_selector";
    let all = KubernetesDiscoveryConfig {
        encode_selector: selector(),
        prefill_selector: selector(),
        decode_selector: selector(),
        ..Default::default()
    };
    assert!(DiscoveryConfig::from(all.clone()).validate(roles).is_ok());
    for missing in [
        KubernetesDiscoveryConfig {
            encode_selector: HashMap::new(),
            ..all.clone()
        },
        KubernetesDiscoveryConfig {
            prefill_selector: HashMap::new(),
            ..all.clone()
        },
        KubernetesDiscoveryConfig {
            decode_selector: HashMap::new(),
            ..all
        },
    ] {
        assert_refused(missing, roles, reason);
    }
}

#[test]
fn a_port_of_zero_is_refused() {
    for (config, field) in [
        (
            DiscoveryConfig::from(KubernetesDiscoveryConfig {
                port: 0,
                ..kubernetes()
            }),
            "discovery.port",
        ),
        (
            SlurmDiscoveryConfig {
                port: 0,
                ..regular_slurm()
            }
            .into(),
            "discovery.port",
        ),
        (
            SlurmDiscoveryConfig {
                bootstrap_port: Some(0),
                ..slurm([&[], &["encode"], &["prefill"], &["decode"]])
            }
            .into(),
            "discovery.bootstrap_port",
        ),
    ] {
        assert_invalid(config, field, "0", "Port must be > 0");
    }
}

#[test]
fn test_validate_invalid_kv_annotation_names() {
    let reason = "Annotation name must not be empty or padded with whitespace";
    for bad in ["", " ", " example.com/key", "example.com/key "] {
        assert_invalid(
            KubernetesDiscoveryConfig {
                kv_connector_annotation: bad.into(),
                ..kubernetes()
            },
            "discovery.kv_connector_annotation",
            bad,
            reason,
        );
        assert_invalid(
            KubernetesDiscoveryConfig {
                kv_engine_id_annotation: bad.into(),
                ..kubernetes()
            },
            "discovery.kv_engine_id_annotation",
            bad,
            reason,
        );
        assert_invalid(
            KubernetesDiscoveryConfig {
                pairing_protocol_label: bad.into(),
                ..kubernetes()
            },
            "discovery.pairing_protocol_label",
            bad,
            "Label name must not be empty or padded with whitespace",
        );
    }
}

#[test]
fn slurm_refuses_an_empty_or_comma_separated_job_name() {
    let reason = "Must be non-empty and contain no comma";
    for bad in ["", "first,second"] {
        for (config, field) in [
            (slurm([&[bad], &[], &[], &[]]), "discovery.job_names"),
            (
                slurm([&[], &[bad], &["p"], &["d"]]),
                "discovery.encode_job_names",
            ),
            (
                slurm([&[], &[], &[bad], &[]]),
                "discovery.prefill_job_names",
            ),
            (slurm([&[], &[], &[], &[bad]]), "discovery.decode_job_names"),
        ] {
            assert_invalid(config, field, bad, reason);
        }
    }
}

#[test]
fn slurm_refuses_an_empty_or_comma_separated_scope() {
    let reason = "Must be non-empty and contain no comma";
    for bad in ["", "first,second"] {
        for (config, field) in [
            (
                SlurmDiscoveryConfig {
                    user: Some(bad.into()),
                    ..regular_slurm()
                },
                "discovery.user",
            ),
            (
                SlurmDiscoveryConfig {
                    account: Some(bad.into()),
                    ..regular_slurm()
                },
                "discovery.account",
            ),
            (
                SlurmDiscoveryConfig {
                    partition: Some(bad.into()),
                    ..regular_slurm()
                },
                "discovery.partition",
            ),
        ] {
            assert_invalid(config, field, bad, reason);
        }
    }
}

#[test]
fn slurm_refuses_a_job_name_shared_by_two_roles() {
    assert_refused(
        slurm([&[], &[], &["shared"], &["shared"]]),
        Roles::PrefillDecode,
        "Slurm job name 'shared' maps to both prefill and decode workers",
    );
}

#[test]
fn test_validate_slurm_discovery() {
    for (roles, config) in [
        (
            Roles::Regular,
            slurm([&["worker", "worker"], &[], &[], &[]]),
        ),
        (Roles::PrefillDecode, slurm([&[], &[], &["prefill"], &[]])),
        (Roles::PrefillDecode, slurm([&[], &[], &[], &["decode"]])),
        (
            Roles::EncodePrefillDecode,
            SlurmDiscoveryConfig {
                bootstrap_port: Some(8998),
                ..slurm([&[], &["encode"], &["prefill"], &["decode"]])
            },
        ),
    ] {
        let config = DiscoveryConfig::from(config);
        assert!(config.validate(roles).is_ok(), "{config:?}");
    }
}

#[test]
fn slurm_requires_a_regular_job_name_for_regular_roles() {
    for prefill in [&[][..], &["prefill"]] {
        assert_refused(
            slurm([&[], &[], prefill, &[]]),
            Roles::Regular,
            "Regular mode with Slurm discovery requires a regular job name",
        );
    }
}

#[test]
fn slurm_refuses_disaggregated_job_names_for_regular_roles() {
    for names in [
        [&["w"][..], &["e"], &[], &[]],
        [&["w"], &[], &["p"], &[]],
        [&["w"], &[], &[], &["d"]],
    ] {
        assert_refused(
            slurm(names),
            Roles::Regular,
            "Regular mode has no encode, prefill or decode workers to map Slurm jobs to",
        );
    }
}

#[test]
fn slurm_requires_a_prefill_or_decode_job_name_for_the_pair() {
    assert_refused(
        slurm([&[], &[], &[], &[]]),
        Roles::PrefillDecode,
        "PD mode with Slurm discovery requires a prefill or decode job name",
    );
}

#[test]
fn slurm_refuses_regular_and_encode_job_names_for_the_pair() {
    for names in [[&["w"][..], &[], &["p"], &[]], [&[], &["e"], &[], &["d"]]] {
        assert_refused(
            slurm(names),
            Roles::PrefillDecode,
            "PD mode has no regular or encode workers to map Slurm jobs to",
        );
    }
}

#[test]
fn slurm_requires_a_job_name_for_each_of_the_three_roles() {
    for names in [
        [&[][..], &[], &["p"], &["d"]],
        [&[], &["e"], &[], &["d"]],
        [&[], &["e"], &["p"], &[]],
    ] {
        assert_refused(
            slurm(names),
            Roles::EncodePrefillDecode,
            "EPD mode with Slurm discovery requires encode, prefill and decode job names",
        );
    }
}

#[test]
fn slurm_refuses_regular_job_names_for_the_three_roles() {
    assert_refused(
        slurm([&["w"], &["e"], &["p"], &["d"]]),
        Roles::EncodePrefillDecode,
        "EPD mode has no regular workers to map Slurm jobs to",
    );
}

#[test]
fn slurm_requires_an_encode_or_prefill_job_name_for_a_bootstrap_port() {
    assert_refused(
        SlurmDiscoveryConfig {
            bootstrap_port: Some(8998),
            ..slurm([&[], &[], &[], &["decode"]])
        },
        Roles::PrefillDecode,
        "A Slurm bootstrap port applies only to encode and prefill workers",
    );
}

#[test]
fn test_validate_file_discovery() {
    assert_invalid(
        FileDiscoveryConfig {
            path: Default::default(),
            ..file()
        },
        "discovery.path",
        "",
        "File discovery needs a manifest path",
    );
}

#[test]
fn test_validate_consul_discovery() {
    assert_invalid(
        ConsulDiscoveryConfig {
            service: String::new(),
            ..consul()
        },
        "discovery.service",
        "",
        "Consul discovery needs a service name",
    );
}

#[test]
fn consul_refuses_an_empty_address_or_datacenter_when_given() {
    let reason = "Must not be empty when given";
    assert_invalid(
        ConsulDiscoveryConfig {
            address: Some(String::new()),
            ..consul()
        },
        "discovery.address",
        "",
        reason,
    );
    assert_invalid(
        ConsulDiscoveryConfig {
            datacenter: Some(String::new()),
            ..consul()
        },
        "discovery.datacenter",
        "",
        reason,
    );
}

#[test]
fn every_provider_requires_a_positive_check_interval() {
    for config in [
        DiscoveryConfig::from(KubernetesDiscoveryConfig {
            check_interval_secs: 0,
            ..kubernetes()
        }),
        DiscoveryConfig::from(FileDiscoveryConfig {
            check_interval_secs: 0,
            ..file()
        }),
        DiscoveryConfig::from(SlurmDiscoveryConfig {
            check_interval_secs: 0,
            ..regular_slurm()
        }),
        DiscoveryConfig::from(ConsulDiscoveryConfig {
            check_interval_secs: 0,
            ..consul()
        }),
    ] {
        assert_invalid(config, "discovery.check_interval_secs", "0", "Must be > 0");
    }
}
