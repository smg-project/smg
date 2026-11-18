use super::*;

fn policy() -> PolicyConfig {
    serde_json::from_value(serde_json::json!({"type": "prefill_cost"})).unwrap()
}

fn config() -> RouterConfig {
    RouterConfig {
        policy: policy(),
        connection_mode: ConnectionMode::Grpc,
        mode: RoutingMode::Regular {
            worker_urls: vec!["grpc://worker:8080".into()],
        },
        ..RouterConfig::default()
    }
}

#[test]
fn prefill_cost_is_a_distinct_roundtrippable_policy_with_its_own_defaults() {
    let policy = policy();
    assert_eq!(policy.name(), "prefill_cost");
    assert!(matches!(
        policy,
        PolicyConfig::PrefillCost {
            block_size: 16,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
            kv_pressure_weight: 0.15
        }
    ));
    let decoded: PolicyConfig =
        serde_json::from_value(serde_json::to_value(&policy).unwrap()).unwrap();
    assert_eq!(decoded.name(), "prefill_cost");
    // Preserve the low-level Rust default; the CLI's cache_aware default is tested separately.
    assert_eq!(RouterConfig::default().policy.name(), "random");
}

#[test]
fn prefill_cost_accepts_regular_grpc_and_rejects_other_transports_and_pd() {
    let mut config = config();
    ConfigValidator::validate_prefill_cost_mode(&config).unwrap();
    config.enable_igw = true;
    assert!(ConfigValidator::validate_prefill_cost_mode(&config).is_err());
    // IGW can discover HTTP workers and derive role policies later even with
    // an empty regular native-gRPC startup list.
    config.mode = RoutingMode::Regular {
        worker_urls: vec![],
    };
    assert!(ConfigValidator::validate_prefill_cost_mode(&config).is_err());
    config.enable_igw = false;
    ConfigValidator::validate_prefill_cost_mode(&config).unwrap();
    config.connection_mode = ConnectionMode::Http;
    assert!(ConfigValidator::validate_prefill_cost_mode(&config)
        .unwrap_err()
        .to_string()
        .contains("regular native gRPC"));
    config.connection_mode = ConnectionMode::Grpc;
    config.mode = RoutingMode::Regular {
        worker_urls: vec!["http://worker:8080".into()],
    };
    assert!(ConfigValidator::validate_prefill_cost_mode(&config).is_err());
    config.mode = RoutingMode::PrefillDecode {
        prefill_urls: vec![("grpc://prefill:8080".into(), None)],
        decode_urls: vec!["grpc://decode:8080".into()],
        prefill_policy: None,
        decode_policy: None,
    };
    assert!(ConfigValidator::validate_prefill_cost_mode(&config).is_err());
    config.policy = RouterConfig::default().policy;
    if let RoutingMode::PrefillDecode {
        ref mut prefill_policy,
        ..
    } = config.mode
    {
        *prefill_policy = Some(policy());
    }
    assert!(ConfigValidator::validate_prefill_cost_mode(&config).is_err());
}

#[test]
fn prefill_cost_refuses_invalid_rate_weight_or_block_size() {
    for (block_size, weight, mean, rate) in [
        (0, 0.15, 1024, 2000.0),
        (16, f64::NAN, 1024, 2000.0),
        (16, -1.0, 1024, 2000.0),
        (16, 0.15, 0, 2000.0),
        (16, 0.15, 1024, f64::INFINITY),
        (16, 0.15, 1024, 0.0),
    ] {
        assert!(
            ConfigValidator::validate_policy(&PolicyConfig::PrefillCost {
                block_size,
                kv_pressure_weight: weight,
                mean_prefill_tokens: mean,
                default_throughput: rate,
            })
            .is_err()
        );
    }
}
