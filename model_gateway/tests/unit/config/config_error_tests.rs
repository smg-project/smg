use super::ConfigError;

#[test]
fn the_constructor_builds_the_same_error_as_the_literal() {
    let built = ConfigError::invalid_value("port", 0u16, "Must be > 0");
    let literal = ConfigError::InvalidValue {
        field: "port".to_string(),
        value: 0u16.to_string(),
        reason: "Must be > 0".to_string(),
    };
    assert_eq!(built.to_string(), literal.to_string());
    assert_eq!(
        built.to_string(),
        "Invalid value for field 'port': 0 - Must be > 0"
    );
    let ConfigError::InvalidValue {
        field,
        value,
        reason,
    } = built
    else {
        panic!("an invalid value");
    };
    assert_eq!(
        (field.as_str(), value.as_str(), reason.as_str()),
        ("port", "0", "Must be > 0")
    );
}

/// A discovery provider's refusal reaches the caller as the config error
/// of the same kind, in the same words.
#[test]
fn a_discovery_refusal_keeps_its_kind_and_its_words() {
    use smg_service_discovery::DiscoveryConfigError;

    let refusal = DiscoveryConfigError::ValidationFailed {
        reason: "no worker pools".to_string(),
    };
    let words = refusal.to_string();
    let mapped = ConfigError::from(refusal);
    assert!(
        matches!(mapped, ConfigError::ValidationFailed { .. }),
        "{mapped:?}"
    );
    assert_eq!(mapped.to_string(), words);

    let refusal = DiscoveryConfigError::InvalidValue {
        field: "discovery.port".to_string(),
        value: "0".to_string(),
        reason: "Must be > 0".to_string(),
    };
    let words = refusal.to_string();
    let mapped = ConfigError::from(refusal);
    assert!(
        matches!(mapped, ConfigError::InvalidValue { .. }),
        "{mapped:?}"
    );
    assert_eq!(mapped.to_string(), words);
    assert_eq!(
        words,
        "Invalid value for field 'discovery.port': 0 - Must be > 0"
    );
}
