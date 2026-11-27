use super::*;

#[test]
fn defaults_are_inert() {
    let cfg = RlConfig::default();
    assert!(!cfg.enabled);
    assert_eq!(cfg.control_timeout_secs, 600);
    assert_eq!(cfg.fanout_concurrency, 32);
    assert_eq!(cfg.validate(), Ok(()));
}

#[test]
fn serde_roundtrip_and_partial_json() {
    let cfg: RlConfig = serde_json::from_str(r#"{"enabled": true}"#).unwrap();
    assert!(cfg.enabled);
    assert_eq!(cfg.control_timeout_secs, 600);
    let back = serde_json::to_string(&cfg).unwrap();
    assert_eq!(serde_json::from_str::<RlConfig>(&back).unwrap(), cfg);
}

#[test]
fn validate_rejects_zero_values_only_when_enabled() {
    let mut cfg = RlConfig {
        enabled: true,
        control_timeout_secs: 0,
        fanout_concurrency: 32,
    };
    assert!(cfg.validate().unwrap_err().contains("control_timeout_secs"));
    cfg.control_timeout_secs = 600;
    cfg.fanout_concurrency = 0;
    assert!(cfg.validate().unwrap_err().contains("fanout_concurrency"));
    cfg.enabled = false;
    assert_eq!(cfg.validate(), Ok(()));
}
