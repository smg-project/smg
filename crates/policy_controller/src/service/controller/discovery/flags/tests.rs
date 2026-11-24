use std::error::Error;

use clap::Parser;
use smg_service_discovery::RuntimeDiscoveryConfig;
use tempfile::tempdir;

use super::*;
use crate::service::controller::{config::tests::valid, discovery::DiscoverySettings};

#[derive(Parser)]
struct Args {
    #[command(flatten)]
    flags: DiscoveryFlags,
}

fn parse(args: &[&str]) -> Result<DiscoveryFlags, clap::Error> {
    Args::try_parse_from(std::iter::once("controller").chain(args.iter().copied()))
        .map(|args| args.flags)
}

#[test]
fn every_discovery_flag_of_the_gateway_parses_here() {
    for args in [
        &["--service-discovery"][..],
        &["--discovery-provider", "consul"],
        &["--selector", "app=a", "tier=b"],
        &["--service-discovery-port", "8000"],
        &["--service-discovery-namespace", "workers"],
        &["--encode-selector", "role=encode"],
        &["--prefill-selector", "role=prefill"],
        &["--decode-selector", "role=decode"],
        &["--kv-connector-annotation", "example.com/connector"],
        &["--kv-engine-id-annotation", "example.com/worker-id"],
        &["--pairing-protocol-label", "example.com/rollout"],
        &["--model-id-from", "label:model"],
        &["--discovery-file", "/run/workers.json"],
        &["--discovery-check-interval-secs", "5"],
        &["--slurm-job-name", "single", "--slurm-job-name", "other"],
        &["--slurm-encode-job-name", "encode"],
        &["--slurm-prefill-job-name", "prefill"],
        &["--slurm-decode-job-name", "decode"],
        &["--slurm-port", "8000"],
        &["--slurm-bootstrap-port", "8998"],
        &["--slurm-user", "user"],
        &["--slurm-account", "account"],
        &["--slurm-partition", "partition"],
        &["--consul-service", "workers"],
        &["--consul-address", "127.0.0.1:8500"],
        &["--consul-datacenter", "dc"],
        &["--api-key", "worker-key"],
        &["--api-key-file", "/run/key"],
    ] {
        assert!(parse(args).is_ok(), "{args:?}");
    }
    for args in [
        ["--router-selector", "app=a"],
        ["--model-alias", "a=b"],
        ["--model-id-from", "label:"],
        ["--service-discovery-port", "65536"],
        ["--discovery-check-interval-secs", "soon"],
    ] {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}

#[test]
fn the_legacy_flag_and_the_provider_flag_select_the_same_kubernetes_settings(
) -> Result<(), Box<dyn Error>> {
    let options = [
        "--selector",
        "app=worker tier=serve",
        "app=override",
        "--service-discovery-port",
        "8080",
        "--service-discovery-namespace",
        "workers",
        "--encode-selector",
        "role=e",
        "--prefill-selector",
        "role=p",
        "--decode-selector",
        "role=d",
        "--kv-connector-annotation",
        "example.com/connector",
        "--kv-engine-id-annotation",
        "example.com/worker-id",
        "--pairing-protocol-label",
        "example.com/rollout",
        "--model-id-from",
        "label:model",
    ];
    let legacy = parse(&[&["--service-discovery"][..], &options].concat())?.settings()?;
    let named =
        parse(&[&["--discovery-provider", "kubernetes"][..], &options].concat())?.settings()?;
    assert_eq!(legacy, named);
    let Some(DiscoveryConfig::Kubernetes(config)) = named else {
        panic!("expected Kubernetes settings");
    };
    assert_eq!(config.port, 8080);
    assert_eq!(config.namespace.as_deref(), Some("workers"));
    assert_eq!(
        config.selector,
        HashMap::from([
            ("app".into(), "override".into()),
            ("tier".into(), "serve".into())
        ])
    );
    assert_eq!(
        config.encode_selector.get("role").map(String::as_str),
        Some("e")
    );
    assert_eq!(
        config.prefill_selector.get("role").map(String::as_str),
        Some("p")
    );
    assert_eq!(
        config.decode_selector.get("role").map(String::as_str),
        Some("d")
    );
    assert_eq!(config.kv_connector_annotation, "example.com/connector");
    assert_eq!(config.kv_engine_id_annotation, "example.com/worker-id");
    assert_eq!(config.pairing_protocol_label, "example.com/rollout");
    assert_eq!(config.model_id_source.as_deref(), Some("label:model"));
    assert!(parse(&["--service-discovery", "--discovery-provider", "kubernetes"]).is_err());
    Ok(())
}

#[test]
fn a_provider_is_refused_without_what_it_requires() -> Result<(), Box<dyn Error>> {
    for (args, missing) in [
        (&["--discovery-provider", "file"][..], "--discovery-file"),
        (
            &["--discovery-provider", "slurm", "--slurm-port", "8000"],
            "--slurm-job-name",
        ),
        (
            &[
                "--discovery-provider",
                "slurm",
                "--slurm-job-name",
                "worker",
            ],
            "--slurm-port",
        ),
        (&["--discovery-provider", "consul"], "--consul-service"),
    ] {
        let error = parse(args)?
            .settings()
            .expect_err("the provider needs the flag");
        assert!(matches!(error, ConfigError::DiscoveryFlag { flag, .. } if flag == missing));
        assert!(error.to_string().contains(missing));
    }
    Ok(())
}

#[test]
fn a_setting_the_gateway_refuses_is_refused_here_by_its_flag() -> Result<(), Box<dyn Error>> {
    for (args, expected_flag, expected_reason) in [
        (
            &["--discovery-provider", "kubernetes"][..],
            "--discovery-provider",
            "Validation failed: Regular mode with service discovery requires a non-empty selector; the roles follow the prefill, decode and encode selectors and job names given",
        ),
        (
            &["--discovery-provider", "slurm", "--slurm-job-name", "w", "--slurm-port", "8000", "--prefill-selector", "role=p"],
            "--discovery-provider",
            "Validation failed: PD mode with Slurm discovery requires a prefill or decode job name; the roles follow the prefill, decode and encode selectors and job names given",
        ),
        (
            &["--service-discovery", "--selector", "app=worker", "--service-discovery-port", "0"],
            "--service-discovery-port",
            "Invalid value for field 'discovery.port': 0 - Port must be > 0",
        ),
        (
            &["--discovery-provider", "slurm", "--slurm-job-name", "worker", "--slurm-port", "0"],
            "--slurm-port",
            "Invalid value for field 'discovery.port': 0 - Port must be > 0",
        ),
        (
            &["--discovery-provider", "slurm", "--slurm-job-name", "worker", "--slurm-port", "8000", "--slurm-user", "one,two"],
            "--slurm-user",
            "Invalid value for field 'discovery.user': one,two - Must be non-empty and contain no comma",
        ),
        (
            &["--discovery-provider", "slurm", "--slurm-prefill-job-name", "shared", "--slurm-decode-job-name", "shared", "--slurm-port", "8000"],
            "--discovery-provider",
            "Validation failed: Slurm job name 'shared' maps to both prefill and decode workers; the roles follow the prefill, decode and encode selectors and job names given",
        ),
        (
            &["--discovery-provider", "consul", "--consul-service", "workers", "--consul-address", ""],
            "--consul-address",
            "Invalid value for field 'discovery.address':  - Must not be empty when given",
        ),
    ] {
        let result = parse(args)?.configured(|_| None);
        assert!(
            matches!(&result, Err(ConfigError::DiscoveryFlag { flag, reason })
                if *flag == expected_flag && reason == expected_reason),
            "{args:?}: {result:?}"
        );
    }
    Ok(())
}

#[test]
fn the_valid_settings_of_every_provider_still_pass() -> Result<(), Box<dyn Error>> {
    for args in [
        &[
            "--discovery-provider",
            "kubernetes",
            "--encode-selector",
            "role=e",
            "--prefill-selector",
            "role=p",
            "--decode-selector",
            "role=d",
        ][..],
        &[
            "--discovery-provider",
            "file",
            "--discovery-file",
            "workers.json",
        ],
        &[
            "--discovery-provider",
            "slurm",
            "--slurm-decode-job-name",
            "decode",
            "--slurm-port",
            "8000",
        ],
        &[
            "--discovery-provider",
            "consul",
            "--consul-service",
            "workers",
            "--consul-address",
            "http://127.0.0.1:8500",
        ],
    ] {
        assert!(parse(args)?.configured(|_| None)?.is_some(), "{args:?}");
    }
    Ok(())
}

#[test]
fn a_refused_field_uses_the_flag_that_sets_it_for_the_selected_provider(
) -> Result<(), Box<dyn Error>> {
    let cases = [
        (
            &["--service-discovery", "--selector", "app=worker"][..],
            &[
                ("discovery.port", "--service-discovery-port"),
                (
                    "discovery.kv_connector_annotation",
                    "--kv-connector-annotation",
                ),
                (
                    "discovery.kv_engine_id_annotation",
                    "--kv-engine-id-annotation",
                ),
                (
                    "discovery.pairing_protocol_label",
                    "--pairing-protocol-label",
                ),
                ("discovery.model_id_source", "--model-id-from"),
            ][..],
        ),
        (
            &[
                "--discovery-provider",
                "file",
                "--discovery-file",
                "workers.json",
            ],
            &[("discovery.path", "--discovery-file")],
        ),
        (
            &[
                "--discovery-provider",
                "slurm",
                "--slurm-job-name",
                "worker",
                "--slurm-port",
                "8000",
            ],
            &[
                ("discovery.port", "--slurm-port"),
                ("discovery.bootstrap_port", "--slurm-bootstrap-port"),
                ("discovery.job_names", "--slurm-job-name"),
                ("discovery.encode_job_names", "--slurm-encode-job-name"),
                ("discovery.prefill_job_names", "--slurm-prefill-job-name"),
                ("discovery.decode_job_names", "--slurm-decode-job-name"),
                ("discovery.user", "--slurm-user"),
                ("discovery.account", "--slurm-account"),
                ("discovery.partition", "--slurm-partition"),
            ],
        ),
        (
            &[
                "--discovery-provider",
                "consul",
                "--consul-service",
                "workers",
            ],
            &[
                ("discovery.service", "--consul-service"),
                ("discovery.address", "--consul-address"),
                ("discovery.datacenter", "--consul-datacenter"),
                ("CONSUL_HTTP_SSL_VERIFY", "--discovery-provider"),
            ],
        ),
    ];
    for (args, fields) in cases {
        let config = parse(args)?.settings()?.expect("selected provider");
        for &(field, flag) in fields.iter().chain(&[(
            "discovery.check_interval_secs",
            "--discovery-check-interval-secs",
        )]) {
            let error = DiscoveryConfigError::InvalidValue {
                field: field.to_owned(),
                value: "bad-value".to_owned(),
                reason: "This value was refused".to_owned(),
            };
            let displayed = error.to_string();
            assert_eq!(
                refuse_settings(&config, Roles::Regular, error).to_string(),
                format!("{flag}: {displayed}")
            );
        }
        // A provider that fails to build names no field either, and its
        // refusal is none of the roles: it stands in its own words.
        let error = DiscoveryConfigError::ValidationFailed {
            reason: "the provider did not build".to_owned(),
        };
        assert_eq!(
            refuse_settings(&config, Roles::Regular, error).to_string(),
            "--discovery-provider: Validation failed: the provider did not build"
        );
    }
    Ok(())
}

#[test]
fn flags_without_their_providers_are_refused() -> Result<(), Box<dyn Error>> {
    for args in [
        ["--discovery-file", "workers.json"],
        ["--discovery-check-interval-secs", "2"],
        ["--slurm-job-name", "worker"],
        ["--slurm-encode-job-name", "worker"],
        ["--slurm-prefill-job-name", "worker"],
        ["--slurm-decode-job-name", "worker"],
        ["--slurm-port", "8080"],
        ["--slurm-bootstrap-port", "8080"],
        ["--slurm-user", "user"],
        ["--slurm-account", "account"],
        ["--slurm-partition", "partition"],
        ["--consul-service", "worker"],
        ["--consul-address", "127.0.0.1:8500"],
        ["--consul-datacenter", "dc"],
        ["--api-key", "worker-key"],
        ["--api-key-file", "worker-key.txt"],
    ] {
        let error = parse(&args)?
            .settings()
            .expect_err("the flag requires its provider");
        assert!(matches!(error, ConfigError::DiscoveryFlag { flag, .. } if flag == args[0]));
    }
    assert!(
        parse(&["--discovery-provider", "file", "--consul-service", "worker"])?
            .settings()
            .is_err()
    );
    let error = parse(&[
        "--service-discovery",
        "--discovery-check-interval-secs",
        "0",
    ])?
    .settings()
    .expect_err("a zero interval is refused before the provider's settings");
    assert!(matches!(
        error,
        ConfigError::DiscoveryFlag {
            flag: "--discovery-check-interval-secs",
            ..
        }
    ));
    Ok(())
}

#[test]
fn the_check_interval_reaches_every_provider() -> Result<(), Box<dyn Error>> {
    for args in [
        &["--service-discovery", "--selector", "app=worker"][..],
        &[
            "--discovery-provider",
            "file",
            "--discovery-file",
            "/run/workers.json",
        ],
        &[
            "--discovery-provider",
            "slurm",
            "--slurm-encode-job-name",
            "encode",
            "--slurm-prefill-job-name",
            "prefill",
            "--slurm-decode-job-name",
            "decode",
            "--slurm-port",
            "8000",
            "--slurm-bootstrap-port",
            "8998",
            "--slurm-user",
            "user",
            "--slurm-account",
            "account",
            "--slurm-partition",
            "partition",
        ],
        &[
            "--discovery-provider",
            "consul",
            "--consul-service",
            "worker",
            "--consul-address",
            "127.0.0.1:8500",
            "--consul-datacenter",
            "dc",
        ],
    ] {
        let flags = parse(&[args, &["--discovery-check-interval-secs", "13"]].concat())?;
        let interval = match flags.settings()?.expect("selected provider") {
            DiscoveryConfig::Kubernetes(config) => config.check_interval_secs,
            DiscoveryConfig::File(config) => {
                assert_eq!(config.path, Path::new("/run/workers.json"));
                config.check_interval_secs
            }
            DiscoveryConfig::Slurm(config) => {
                assert!(config.job_names.is_empty());
                assert_eq!(config.encode_job_names, ["encode"]);
                assert_eq!(config.prefill_job_names, ["prefill"]);
                assert_eq!(config.decode_job_names, ["decode"]);
                assert_eq!(config.port, 8000);
                assert_eq!(config.bootstrap_port, Some(8998));
                assert_eq!(config.user.as_deref(), Some("user"));
                assert_eq!(config.account.as_deref(), Some("account"));
                assert_eq!(config.partition.as_deref(), Some("partition"));
                config.check_interval_secs
            }
            DiscoveryConfig::Consul(config) => {
                assert_eq!(config.service, "worker");
                assert_eq!(config.address.as_deref(), Some("127.0.0.1:8500"));
                assert_eq!(config.datacenter.as_deref(), Some("dc"));
                config.check_interval_secs
            }
        };
        assert_eq!(interval, 13, "{args:?}");
    }
    let flags = parse(&[
        "--discovery-provider",
        "slurm",
        "--slurm-port",
        "8000",
        "--slurm-job-name",
        "worker",
        "--slurm-job-name",
        "another",
        "--discovery-check-interval-secs",
        "13",
    ])?;
    let Some(DiscoveryConfig::Slurm(config)) = flags.settings()? else {
        panic!("expected Slurm settings");
    };
    assert_eq!(config.job_names, ["worker", "another"]);
    assert_eq!(config.check_interval_secs, 13);
    Ok(())
}

#[test]
fn with_no_discovery_flag_there_are_no_settings() -> Result<(), Box<dyn Error>> {
    assert_eq!(parse(&[])?.settings()?, None);
    assert_eq!(parse(&["--selector", "app=worker"])?.settings()?, None);
    assert!(parse(&["--selector", "app=worker missing-equals"])?
        .settings()
        .is_err());
    assert!(parse(&["--selector", "=missing-key"])?.settings().is_err());
    assert!(parse(&["--selector", " "])?.settings().is_err());
    let file = [
        "--discovery-provider",
        "file",
        "--discovery-file",
        "workers.json",
    ];
    assert_eq!(
        parse(&[&file[..], &["--selector", "app=worker"]].concat())?.settings()?,
        parse(&file)?.settings()?
    );
    Ok(())
}

#[test]
fn the_roles_follow_the_selectors_and_job_names_given() -> Result<(), Box<dyn Error>> {
    assert_eq!(parse(&[])?.roles(), None);
    assert_eq!(parse(&["--encode-selector", "role=e"])?.roles(), None);
    assert_eq!(parse(&["--slurm-job-name", "worker"])?.roles(), None);
    for (role, encoder) in [
        ("--prefill-selector", "--encode-selector"),
        ("--decode-selector", "--encode-selector"),
        ("--slurm-prefill-job-name", "--slurm-encode-job-name"),
        ("--slurm-decode-job-name", "--slurm-encode-job-name"),
    ] {
        assert_eq!(
            parse(&[role, "role=p"])?.roles(),
            Some(Roles::PrefillDecode)
        );
        assert_eq!(
            parse(&[role, "role=p", encoder, "role=e"])?.roles(),
            Some(Roles::EncodePrefillDecode)
        );
    }
    Ok(())
}

#[test]
fn the_defaults_are_the_gateways() -> Result<(), Box<dyn Error>> {
    let Some(DiscoveryConfig::Kubernetes(config)) =
        parse(&["--service-discovery", "--selector", "app=worker"])?.settings()?
    else {
        panic!("expected Kubernetes settings");
    };
    assert_eq!(config.port, 80);
    assert_eq!(config.kv_connector_annotation, "smg.ai/kv-connector");
    assert_eq!(config.kv_engine_id_annotation, "smg.ai/kv-engine-id");
    assert_eq!(config.pairing_protocol_label, "smg.ai/pairing-protocol");
    assert_eq!(config.check_interval_secs, 30);
    Ok(())
}

#[test]
fn the_key_is_read_only_for_a_selected_provider() -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    // A key file that is not there fails every read of the key.
    let missing = dir.path().join("missing");
    let env = |name: &str| match name {
        "SMG_API_KEY_FILE" => Some(missing.clone().into_os_string()),
        _ => None,
    };
    assert_eq!(parse(&[])?.configured(env)?, None);
    assert_eq!(parse(&["--selector", "app=worker"])?.configured(env)?, None);
    let error = parse(&[
        "--discovery-provider",
        "file",
        "--discovery-file",
        "workers.json",
    ])?
    .configured(env)
    .expect_err("the provider's key is read");
    assert!(matches!(
        error,
        ConfigError::DiscoveryFlag {
            flag: "--api-key-file",
            ..
        }
    ));
    Ok(())
}

#[test]
fn the_key_is_read_from_the_flag_the_environment_or_its_file_and_never_prints(
) -> Result<(), Box<dyn Error>> {
    let dir = tempdir()?;
    let path = dir.path().join("key");
    let file = path.to_str().expect("a UTF-8 path");
    fs::write(&path, "  worker-file-secret\n")?;
    let from_env = |name: &str| match name {
        "SMG_API_KEY" => Some("  worker-env-secret\n".into()),
        _ => None,
    };
    let file_env = |name: &str| match name {
        "SMG_API_KEY_FILE" => Some(path.clone().into_os_string()),
        _ => None,
    };
    for (args, env, expected) in [
        (
            &["--api-key", "worker-flag-secret"][..],
            &from_env as &dyn Fn(&str) -> Option<OsString>,
            "worker-flag-secret",
        ),
        (&[][..], &from_env, "worker-env-secret"),
        (
            &["--api-key-file", file][..],
            &|_| None,
            "worker-file-secret",
        ),
        (&[][..], &file_env, "worker-file-secret"),
    ] {
        let provider = [
            "--discovery-provider",
            "file",
            "--discovery-file",
            "workers.json",
        ];
        let flags = parse(&[&provider[..], args].concat())?;
        let key = flags.credential_with_env(env)?.expect("key");
        assert_eq!(key.value(), expected);
        let flags_debug = format!("{flags:?}");
        assert!(
            !flags_debug.contains(expected),
            "flag debug exposed the key"
        );
        if args.first() == Some(&"--api-key") {
            assert!(flags_debug.contains("<redacted>"));
        }
        let mut config = valid();
        config.discovery = flags
            .settings()?
            .map(|settings| DiscoverySettings::new(settings, flags.roles(), Some(key)))
            .transpose()?;
        assert!(matches!(
            config.discovery.as_ref().map(DiscoverySettings::runtime),
            Some(RuntimeDiscoveryConfig::File(_))
        ));
        let config_debug = format!("{config:?}");
        assert!(config_debug.contains("<redacted>"));
        assert!(
            !config_debug.contains(expected),
            "configuration debug exposed the key"
        );
    }

    for error in [
        parse(&["--api-key", "worker-flag-secret", "--api-key-file", file])?
            .credential_with_env(|_| None),
        parse(&["--api-key-file", file])?.credential_with_env(from_env),
        parse(&["--api-key", "worker-flag-secret"])?.credential_with_env(file_env),
    ] {
        let error = error.expect_err("the key and the key file conflict");
        let text = error.to_string();
        assert!(text.contains("--api-key"));
        assert!(text.contains("--api-key-file"));
        assert!(!text.contains("worker-flag-secret"));
        assert!(!text.contains("worker-env-secret"));
    }

    for text in ["  \n", "first\nsecond\n"] {
        fs::write(&path, text)?;
        let error = parse(&["--api-key-file", file])?
            .credential_with_env(|_| None)
            .expect_err("the file needs one key");
        assert!(error.to_string().contains("--api-key-file"));
        assert!(!error.to_string().contains("first"));
    }
    assert!(parse(&["--api-key", " "])?
        .credential_with_env(|_| None)
        .is_err());
    assert!(parse(&[])?.credential_with_env(|_| None)?.is_none());
    Ok(())
}
