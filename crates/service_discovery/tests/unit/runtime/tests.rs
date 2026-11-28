use super::*;

#[cfg(any(
    feature = "consul",
    feature = "file",
    feature = "kubernetes",
    feature = "slurm"
))]
fn assert_refuses_as_validate(config: DiscoveryConfig, roles: Roles) {
    let runtime = RuntimeDiscoveryConfig::from_config(&config, Some(roles));
    assert!(runtime.is_err(), "{runtime:?}");
    assert_eq!(
        runtime.map(|_| ()).map_err(|error| error.to_string()),
        config.validate(roles).map_err(|error| error.to_string())
    );
}

#[cfg(feature = "kubernetes")]
#[test]
fn from_config_refuses_missing_selectors_before_resolving_kubernetes_settings() {
    assert_refuses_as_validate(
        KubernetesDiscoveryConfig {
            model_id_source: Some("pod-name".into()),
            ..Default::default()
        }
        .into(),
        Roles::Regular,
    );
}

#[cfg(feature = "file")]
#[test]
fn from_config_refuses_an_empty_manifest_path() {
    assert_refuses_as_validate(
        FileDiscoveryConfig {
            path: Default::default(),
            ..file_settings()
        }
        .into(),
        Roles::Regular,
    );
}

#[cfg(feature = "slurm")]
#[test]
fn from_config_refuses_a_zero_slurm_port() {
    assert_refuses_as_validate(
        SlurmDiscoveryConfig {
            port: 0,
            ..slurm_settings()
        }
        .into(),
        Roles::Regular,
    );
}

#[cfg(feature = "consul")]
#[test]
fn from_config_refuses_a_zero_interval_before_resolving_consul_settings() {
    let config = ConsulDiscoveryConfig {
        service: "workers".into(),
        address: Some("http://".into()),
        datacenter: None,
        check_interval_secs: 0,
    };
    assert_refuses_as_validate(config.into(), Roles::Regular);
}

#[cfg(feature = "kubernetes")]
#[test]
fn without_roles_kubernetes_checks_settings_that_apply_to_any_roles() {
    let config = KubernetesDiscoveryConfig::default();
    assert!(matches!(
        RuntimeDiscoveryConfig::from_config(&config.clone().into(), None),
        Ok(RuntimeDiscoveryConfig::Kubernetes(runtime)) if !runtime.disaggregated_mode
    ));

    let config = KubernetesDiscoveryConfig { port: 0, ..config }.into();
    let error = RuntimeDiscoveryConfig::from_config(&config, None)
        .expect_err("the worker port must be positive");
    assert_eq!(
        error.to_string(),
        "Invalid value for field 'discovery.port': 0 - Port must be > 0"
    );
}

#[cfg(any(feature = "consul", feature = "file", feature = "slurm"))]
#[test]
fn file_slurm_and_consul_need_worker_roles() {
    // One call a provider, not a table: a build with one of the three
    // would loop over a single row.
    fn needs_roles(config: &DiscoveryConfig, expected: &str) {
        let result = RuntimeDiscoveryConfig::from_config(config, None);
        assert!(
            matches!(&result, Err(DiscoveryConfigError::ValidationFailed { reason }) if reason == expected),
            "{result:?}"
        );
    }
    #[cfg(feature = "file")]
    needs_roles(
        &file_settings().into(),
        "File discovery needs regular, prefill-decode or encode-prefill-decode routing",
    );
    #[cfg(feature = "slurm")]
    needs_roles(
        &slurm_settings().into(),
        "Slurm discovery needs regular, prefill-decode or encode-prefill-decode routing",
    );
    #[cfg(feature = "consul")]
    needs_roles(
        &consul_settings().into(),
        "Consul discovery needs regular, prefill-decode or encode-prefill-decode routing",
    );
}

#[cfg(feature = "slurm")]
#[test]
fn without_roles_slurm_checks_job_names_before_resolving_settings() {
    let config = SlurmDiscoveryConfig {
        job_names: vec!["first,second".into()],
        ..slurm_settings()
    }
    .into();
    let error = RuntimeDiscoveryConfig::from_config(&config, None)
        .expect_err("a job name must be one filter");
    assert_eq!(
        error.to_string(),
        "Invalid value for field 'discovery.job_names': first,second - Must be non-empty and contain no comma"
    );
}

/// Every Kubernetes setting reaches the runtime, the KV annotation names
/// included: those once flowed through only one of two hand-written
/// conversions.
#[cfg(feature = "kubernetes")]
#[test]
fn kubernetes_settings_reach_the_runtime() {
    let config = DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
        namespace: Some("prod".to_string()),
        port: 9000,
        check_interval_secs: 45,
        selector: [("app".to_string(), "worker".to_string())].into(),
        bootstrap_port_annotation: "example.com/bootstrap".to_string(),
        worker_ports_annotation: "example.com/ports".to_string(),
        kv_connector_annotation: "example.com/connector".to_string(),
        kv_engine_id_annotation: "example.com/engine-id".to_string(),
        pairing_protocol_label: "example.com/rollout".to_string(),
        model_id_source: Some("namespace".to_string()),
        ..Default::default()
    });

    let Ok(RuntimeDiscoveryConfig::Kubernetes(runtime)) =
        RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular))
    else {
        panic!("expected Kubernetes runtime discovery");
    };

    assert_eq!(runtime.namespace.as_deref(), Some("prod"));
    assert_eq!(runtime.port, 9000);
    assert_eq!(runtime.check_interval, Duration::from_secs(45));
    assert_eq!(
        runtime.selector.get("app").map(String::as_str),
        Some("worker")
    );
    assert_eq!(runtime.bootstrap_port_annotation, "example.com/bootstrap");
    assert_eq!(runtime.worker_ports_annotation, "example.com/ports");
    assert_eq!(runtime.kv_connector_annotation, "example.com/connector");
    assert_eq!(runtime.kv_engine_id_annotation, "example.com/engine-id");
    assert_eq!(runtime.pairing_protocol_label, "example.com/rollout");
    assert!(matches!(
        runtime.model_id_source,
        Some(ModelIdSource::Namespace)
    ));
    assert!(!runtime.disaggregated_mode);
}

#[cfg(feature = "kubernetes")]
#[test]
fn disaggregated_modes_select_role_selectors() {
    let selector = || [("role".into(), "worker".into())].into();
    let config = KubernetesDiscoveryConfig {
        encode_selector: selector(),
        prefill_selector: selector(),
        decode_selector: selector(),
        ..Default::default()
    }
    .into();
    for roles in [Roles::PrefillDecode, Roles::EncodePrefillDecode] {
        assert!(
            matches!(
                RuntimeDiscoveryConfig::from_config(&config, Some(roles)),
                Ok(RuntimeDiscoveryConfig::Kubernetes(runtime)) if runtime.disaggregated_mode
            ),
            "{roles:?}"
        );
    }
}

fn file_settings() -> FileDiscoveryConfig {
    FileDiscoveryConfig {
        path: "/run/smg/workers.json".into(),
        check_interval_secs: 15,
    }
}

#[cfg(feature = "file")]
#[test]
fn file_settings_reach_the_runtime() {
    let config = file_settings().into();
    let Ok(RuntimeDiscoveryConfig::File(runtime)) =
        RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular))
    else {
        panic!("expected file runtime discovery");
    };
    assert_eq!(runtime.path, std::path::Path::new("/run/smg/workers.json"));
    assert_eq!(runtime.check_interval, Duration::from_secs(15));
}

fn slurm_settings() -> SlurmDiscoveryConfig {
    SlurmDiscoveryConfig {
        job_names: vec!["smg-worker".to_string()],
        encode_job_names: vec![],
        prefill_job_names: vec![],
        decode_job_names: vec![],
        port: 8000,
        bootstrap_port: None,
        user: None,
        account: None,
        partition: None,
        check_interval_secs: 10,
    }
}

#[cfg(feature = "slurm")]
#[test]
fn slurm_settings_reach_the_runtime() {
    let config = slurm_settings().into();
    let Ok(RuntimeDiscoveryConfig::Slurm(runtime)) =
        RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular))
    else {
        panic!("expected Slurm runtime discovery");
    };
    assert_eq!(runtime.check_interval.as_secs(), 10);
}

fn consul_settings() -> ConsulDiscoveryConfig {
    ConsulDiscoveryConfig {
        service: "smg-worker".to_string(),
        address: Some("http://consul.internal:8500".to_string()),
        datacenter: Some("dc2".to_string()),
        check_interval_secs: 10,
    }
}

#[cfg(feature = "consul")]
#[test]
fn consul_settings_reach_the_runtime() {
    let config = consul_settings().into();
    let Ok(RuntimeDiscoveryConfig::Consul(runtime)) =
        RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular))
    else {
        panic!("expected Consul runtime discovery");
    };
    assert_eq!(runtime.check_interval.as_secs(), 10);
}

/// What a build says of settings for a provider it does not hold: the
/// provider's name and the feature that builds it.
#[cfg(not(all(
    feature = "consul",
    feature = "file",
    feature = "kubernetes",
    feature = "slurm"
)))]
fn assert_unavailable(config: DiscoveryConfig, expected: &str) {
    let result = RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular));
    assert_eq!(
        result.map(|_| ()).map_err(|error| error.to_string()),
        Err(expected.to_owned())
    );
}

#[cfg(not(feature = "kubernetes"))]
#[test]
fn a_build_without_the_kubernetes_provider_refuses_its_settings() {
    assert_unavailable(
        KubernetesDiscoveryConfig::default().into(),
        "Validation failed: this build holds no Kubernetes discovery provider: build `smg-service-discovery` with its `kubernetes` feature"
    );
}

#[cfg(not(feature = "file"))]
#[test]
fn a_build_without_the_file_provider_refuses_its_settings() {
    assert_unavailable(
        FileDiscoveryConfig { path: Default::default(), ..file_settings() }.into(),
        "Validation failed: this build holds no file discovery provider: build `smg-service-discovery` with its `file` feature"
    );
}

#[cfg(not(feature = "slurm"))]
#[test]
fn a_build_without_the_slurm_provider_refuses_its_settings() {
    assert_unavailable(
        SlurmDiscoveryConfig { port: 0, ..slurm_settings() }.into(),
        "Validation failed: this build holds no Slurm discovery provider: build `smg-service-discovery` with its `slurm` feature"
    );
}

#[cfg(not(feature = "consul"))]
#[test]
fn a_build_without_the_consul_provider_refuses_its_settings() {
    assert_unavailable(
        ConsulDiscoveryConfig { check_interval_secs: 0, ..consul_settings() }.into(),
        "Validation failed: this build holds no Consul discovery provider: build `smg-service-discovery` with its `consul` feature"
    );
}

#[cfg(feature = "kubernetes")]
#[test]
fn invalid_model_id_source_is_a_config_error() {
    let config = DiscoveryConfig::Kubernetes(KubernetesDiscoveryConfig {
        selector: [("app".into(), "worker".into())].into(),
        model_id_source: Some("pod-name".to_string()),
        ..Default::default()
    });
    let err = RuntimeDiscoveryConfig::from_config(&config, Some(Roles::Regular))
        .expect_err("invalid model ID source");
    assert!(
        matches!(err, DiscoveryConfigError::InvalidValue { ref field, .. } if field == "discovery.model_id_source"),
        "{err}"
    );
}
