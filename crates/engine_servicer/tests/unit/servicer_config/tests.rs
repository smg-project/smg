use std::time::Duration;

use super::ServicerConfig;
use crate::{
    ServicerError, DEFAULT_ENGINE_PREFILL_TOKENS_PER_SEC, DEFAULT_ENGINE_SILENCE_TIMEOUT,
    DEFAULT_ENGINE_STARTUP_CEILING, DEFAULT_ENGINE_STARTUP_TIMEOUT,
};

fn valid() -> ServicerConfig {
    ServicerConfig::new("127.0.0.1:0", "ipc:///tmp/engine", "tcp://127.0.0.1:1")
}

/// What a lifecycle owner gets when it says nothing beyond the endpoints:
/// one engine, no tokenizer directory, the crate's start-up bounds and
/// its engine-silence bound.
#[test]
fn a_new_config_keeps_the_endpoints_and_carries_the_crate_defaults() {
    let config = valid();
    assert_eq!(config.bind_address, "127.0.0.1:0");
    assert_eq!(config.ipc_base_url, "ipc:///tmp/engine");
    assert_eq!(config.handshake_address, "tcp://127.0.0.1:1");
    assert_eq!(config.engine_count, 1);
    assert_eq!(config.tokenizer_dir, None);
    assert_eq!(
        config.engine_startup_timeout,
        DEFAULT_ENGINE_STARTUP_TIMEOUT
    );
    assert_eq!(
        config.engine_startup_ceiling,
        Some(DEFAULT_ENGINE_STARTUP_CEILING)
    );
    assert_eq!(
        config.engine_silence_timeout,
        DEFAULT_ENGINE_SILENCE_TIMEOUT
    );
    assert_eq!(config.engine_silence_timeout, Duration::from_secs(300));
    assert_eq!(
        config.engine_prefill_tokens_per_sec,
        DEFAULT_ENGINE_PREFILL_TOKENS_PER_SEC
    );
    let bounds = config.silence_bounds();
    assert_eq!(bounds.window, Duration::from_secs(300));
    assert_eq!(
        bounds.prefill_tokens_per_sec,
        DEFAULT_ENGINE_PREFILL_TOKENS_PER_SEC
    );
    assert_eq!(config.metrics_bind_address, None);
    config.validate().expect("the defaults are valid");
}

/// No prefill allowance (`0` tokens per second) is a valid choice: the
/// bound is then the silence window alone.
#[test]
fn no_prefill_allowance_is_valid() {
    let mut config = valid();
    config.engine_prefill_tokens_per_sec = 0;
    config.validate().expect("valid");
}

/// A metrics listener is opt-in: a `host:port` of its own, on a port
/// other than the gRPC one (two ephemeral ports are two ports).
#[test]
fn a_metrics_bind_address_is_optional_and_its_own() {
    let mut config = valid();
    config.metrics_bind_address = Some("127.0.0.1:0".to_string());
    config.validate().expect("valid: two ephemeral ports");
    config.metrics_bind_address = Some("[::]:9000".to_string());
    config.validate().expect("valid");
}

/// An IPC handshake endpoint (the tests' choice) and no ceiling are valid
/// too.
#[test]
fn an_ipc_handshake_and_no_ceiling_are_valid() {
    let mut config = valid();
    config.handshake_address = "ipc:///tmp/handshake".to_string();
    config.engine_startup_ceiling = None;
    config.validate().expect("valid");
}

/// Each check refuses with the message that names its field, the texts
/// the lifecycle owner shows as they were.
#[test]
fn validate_names_the_field_at_fault() {
    let refusal = |config: ServicerConfig| match config.validate() {
        Err(ServicerError::InvalidConfig(message)) => message,
        other => panic!("expected an invalid config, got {other:?}"),
    };
    let with = |edit: fn(&mut ServicerConfig)| {
        let mut config = valid();
        edit(&mut config);
        config
    };
    assert_eq!(
        refusal(with(|config| config.bind_address = "nope".to_string())),
        "bind_address must be host:port"
    );
    assert_eq!(
        refusal(with(
            |config| config.ipc_base_url = "/tmp/not-ipc".to_string()
        )),
        "ipc_base_url must be ipc://<path>"
    );
    assert_eq!(
        refusal(with(|config| {
            config.handshake_address = "udp://127.0.0.1:1".to_string();
        })),
        "handshake_address must be tcp://host:port or ipc://<path>"
    );
    assert_eq!(
        refusal(with(|config| config.engine_count = 0)),
        "engine_count must be positive"
    );
    assert_eq!(
        refusal(with(|config| config.engine_startup_timeout = Duration::ZERO)),
        "engine_startup_timeout must be positive"
    );
    assert_eq!(
        refusal(with(|config| {
            config.engine_startup_ceiling = Some(Duration::ZERO);
        })),
        "engine_startup_ceiling must be positive when set"
    );
    assert_eq!(
        refusal(with(|config| config.engine_silence_timeout = Duration::ZERO)),
        "engine_silence_timeout must be positive"
    );
    assert_eq!(
        refusal(with(|config| {
            config.metrics_bind_address = Some("nope".to_string());
        })),
        "metrics_bind_address must be host:port"
    );
    assert_eq!(
        refusal(with(|config| {
            config.bind_address = "127.0.0.1:50051".to_string();
            config.metrics_bind_address = Some("127.0.0.1:50051".to_string());
        })),
        "metrics_bind_address must differ from bind_address"
    );
    assert_eq!(
        refusal(with(|config| config.warmup_nonce = Some(String::new()))),
        "warmup_nonce must be ASCII letters and digits"
    );
    assert_eq!(
        refusal(with(|config| config.warmup_nonce = Some("a-b".to_string()))),
        "warmup_nonce must be ASCII letters and digits"
    );
}
