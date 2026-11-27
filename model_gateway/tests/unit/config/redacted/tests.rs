use smg_data_connector::{OracleConfig, PostgresConfig, RedisConfig};

use super::*;
use crate::config::{RoutingMode, TenantApiKeyEntry};

/// Every serialized field of `RouterConfig`, read from its definition:
/// the `pub` fields of the struct that serde does not skip.
fn serialized_fields() -> Vec<String> {
    let source = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/src/config/types.rs"));
    let start = source
        .find("pub struct RouterConfig {")
        .expect("types.rs defines RouterConfig");
    let body = &source[start..];
    let end = body.find("\n}\n").expect("the struct ends");
    let mut fields = Vec::new();
    let mut skipped = false;
    for line in body[..end].lines() {
        let line = line.trim();
        if line.starts_with("#[serde(") {
            skipped |= line.contains("skip)") || line.contains("skip,");
        }
        if let Some((name, _)) = line.strip_prefix("pub ").and_then(|f| f.split_once(':')) {
            if !skipped {
                fields.push(name.to_string());
            }
            skipped = false;
        }
    }
    fields
}

#[test]
fn every_serialized_field_is_classified() {
    let fields = serialized_fields();
    assert!(fields.len() > 100, "the struct was read: {fields:?}");

    let unclassified: Vec<_> = fields
        .iter()
        .filter(|f| !PLAIN.contains(&f.as_str()) && !REDACTED.contains(&f.as_str()))
        .collect();
    assert!(
        unclassified.is_empty(),
        "classify these fields of RouterConfig in config/redacted.rs, plain or redacted: {unclassified:?}"
    );
    let gone: Vec<_> = PLAIN
        .iter()
        .chain(REDACTED)
        .filter(|f| !fields.iter().any(|g| g == *f))
        .collect();
    assert!(
        gone.is_empty(),
        "these classified fields no longer exist on RouterConfig: {gone:?}"
    );
    let both: Vec<_> = PLAIN.iter().filter(|f| REDACTED.contains(f)).collect();
    assert!(
        both.is_empty(),
        "a field is plain or redacted, not both: {both:?}"
    );
}

/// A configuration with a secret in every place one can be.
fn configuration_with_every_secret() -> RouterConfig {
    let mut config = RouterConfig {
        mode: RoutingMode::PrefillDecode {
            prefill_urls: vec![(
                "http://prefill-user:prefill-secret@prefill:8000".to_string(),
                Some(8998),
            )],
            decode_urls: vec!["http://decode-user:decode-secret@decode:8001".to_string()],
            prefill_policy: None,
            decode_policy: None,
        },
        ..RouterConfig::default()
    };
    config.api_key = Some("shared-secret-key".to_string());
    config.tenant_api_keys = vec![TenantApiKeyEntry {
        tenant_id: "team-red".to_string(),
        key: "red-tenant-secret".to_string(),
    }];
    config.postgres = Some(
        serde_json::from_value::<PostgresConfig>(json!({
            "db_url": "postgres://pg-user:pg-secret@db:5432/smg?password=pg-query-secret",
            "pool_max": 4,
        }))
        .expect("a postgres config"),
    );
    config.redis = Some(
        serde_json::from_value::<RedisConfig>(
            json!({ "url": "redis://:redis-secret@cache:6379/0" }),
        )
        .expect("a redis config"),
    );
    config.oracle = Some(
        serde_json::from_value::<OracleConfig>(json!({
            "connect_descriptor": "tcps://db:1522/svc",
            "username": "ora-user",
            "password": "oracle-secret",
        }))
        .expect("an oracle config"),
    );
    config.ca_certificates = vec![b"-----BEGIN CERTIFICATE-----".to_vec()];
    config
}

#[test]
fn the_view_carries_none_of_the_secrets_and_names_each_by_its_mask() {
    let config = configuration_with_every_secret();

    let keys = KeysInForce::of(config.api_key.as_deref(), &config.tenant_api_keys, None);
    let view = redacted(&config, &keys).expect("the view serializes");
    let text = view.to_string();

    for secret in [
        "prefill-secret",
        "decode-secret",
        "shared-secret-key",
        "red-tenant-secret",
        "pg-secret",
        "pg-query-secret",
        "redis-secret",
        "oracle-secret",
        "BEGIN CERTIFICATE",
    ] {
        assert!(!text.contains(secret), "{secret} reached the view: {text}");
    }
    assert_eq!(view["api_key"], json!(key_fingerprint("shared-secret-key")));
    assert_eq!(
        view["tenant_api_keys"],
        json!([{ "tenant_id": "team-red", "fingerprint": key_fingerprint("red-tenant-secret") }])
    );
    assert_eq!(view["mode"]["type"], json!("prefill_decode"));
    assert_eq!(
        view["mode"]["prefill_urls"],
        json!([["http://***@prefill:8000", 8998]])
    );
    assert_eq!(
        view["mode"]["decode_urls"],
        json!(["http://***@decode:8001"])
    );
    assert_eq!(
        view["postgres"]["db_url"],
        json!("postgres://pg-user:****@db:5432/smg?password=****")
    );
    assert_eq!(view["postgres"]["pool_max"], json!(4));
    assert_eq!(view["redis"]["url"], json!("redis://:****@cache:6379/0"));
    assert_eq!(view["oracle"]["username"], json!("ora-user"));
    assert!(
        view["oracle"].get("password").is_none(),
        "the oracle password is left out: {}",
        view["oracle"]
    );
    assert_eq!(view["ca_certificates"], json!(1));
}

#[test]
fn plain_fields_are_copied_as_the_configuration_serializes_them() {
    let config = RouterConfig {
        host: "0.0.0.0".to_string(),
        port: 30123,
        cors_allowed_origins: vec!["https://example.test".to_string()],
        ..RouterConfig::default()
    };

    let view = redacted(&config, &KeysInForce::default()).expect("the view serializes");

    assert_eq!(view["host"], json!("0.0.0.0"));
    assert_eq!(view["port"], json!(30123));
    assert_eq!(
        view["cors_allowed_origins"],
        json!(["https://example.test"])
    );
    assert_eq!(
        view["policy"],
        serde_json::to_value(&config.policy).expect("the policy serializes")
    );
    assert_eq!(view["api_key"], Value::Null, "no key, no fingerprint");
    assert_eq!(view["tenant_api_keys"], json!([]));
}

#[test]
fn the_sources_record_uses_the_start_up_words() {
    let sources = SecretSources {
        api_key: Some(SecretSource::File),
        tenant_api_keys: Some((SecretSource::Environment, 2)),
        control_plane_api_keys: Some((SecretSource::File, 1)),
        postgres_db_url: None,
        redis_url: Some(SecretSource::Flag),
    };

    let keys = KeysInForce {
        tenant_api_keys: vec![
            ("team-red".to_string(), "0".repeat(16)),
            ("team-blue".to_string(), "1".repeat(16)),
        ],
        ..KeysInForce::default()
    };

    assert_eq!(
        secret_sources(&sources, &keys),
        json!({
            "api_key": "file",
            "tenant_api_keys": { "source": "env", "entries": 2 },
            "control_plane_api_keys": { "source": "file", "entries": 1 },
            "postgres_db_url": null,
            "redis_url": "flag",
        })
    );
}

#[test]
fn the_effective_view_has_its_three_parts() {
    let view = effective(
        &RouterConfig::default(),
        &SecretSources::default(),
        &KeysInForce::default(),
    )
    .expect("the view serializes");

    assert_eq!(view["build"]["version"], json!(version::VERSION));
    assert!(view["secret_sources"].is_object(), "{view}");
    assert_eq!(view["config"]["port"], json!(RouterConfig::default().port));
}

#[test]
fn the_view_names_the_keys_in_force_over_the_configuration() {
    let config = RouterConfig {
        api_key: Some("start-key".to_string()),
        ..RouterConfig::default()
    };
    let sources = SecretSources {
        api_key: Some(SecretSource::File),
        tenant_api_keys: Some((SecretSource::File, 1)),
        control_plane_api_keys: Some((SecretSource::File, 1)),
        ..SecretSources::default()
    };
    let keys = KeysInForce::of(
        Some("rotated-key"),
        &[
            TenantApiKeyEntry {
                tenant_id: "team-red".to_string(),
                key: "red-secret".to_string(),
            },
            TenantApiKeyEntry {
                tenant_id: "team-blue".to_string(),
                key: "blue-secret".to_string(),
            },
        ],
        Some(3),
    );

    let view = effective(&config, &sources, &keys).expect("the view serializes");

    assert_eq!(
        view["config"]["api_key"],
        json!(key_fingerprint("rotated-key"))
    );
    assert_eq!(
        view["config"]["tenant_api_keys"],
        json!([
            { "tenant_id": "team-red", "fingerprint": key_fingerprint("red-secret") },
            { "tenant_id": "team-blue", "fingerprint": key_fingerprint("blue-secret") },
        ])
    );
    assert_eq!(
        view["secret_sources"]["tenant_api_keys"]["entries"],
        json!(2)
    );
    assert_eq!(
        view["secret_sources"]["control_plane_api_keys"]["entries"],
        json!(3)
    );
    assert!(
        !view.to_string().contains(&key_fingerprint("start-key")),
        "the start's key is gone from the view: {view}"
    );
}
