use std::collections::HashMap;

use openai_protocol::worker::RuntimeType;

use super::*;

fn labels(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

#[test]
fn sglang_and_vllm_static_rows() {
    let s = capabilities_for(RuntimeType::Sglang, &HashMap::new());
    assert_eq!(s.source, RlCapabilitySource::Static);
    assert_eq!(s.pause_modes, ["abort", "retract", "in_place"]);
    assert_eq!(s.update_from, ["disk", "tensor", "distributed"]);
    assert!(s.abort && s.flush_cache && s.sleep_wake && s.reports_weight_version);

    let v = capabilities_for(RuntimeType::Vllm, &HashMap::new());
    assert_eq!(v.pause_modes, ["abort", "wait", "keep"]);
    assert_eq!(v.update_from, ["disk", "distributed"]);
    assert!(!v.abort && !v.flush_cache && v.sleep_wake && !v.reports_weight_version);
}

#[test]
fn other_runtimes_have_no_capabilities() {
    for rt in [
        RuntimeType::Trtllm,
        RuntimeType::Mlx,
        RuntimeType::Generic,
        RuntimeType::External,
        RuntimeType::Unspecified,
    ] {
        let c = capabilities_for(rt, &HashMap::new());
        assert!(
            c.pause_modes.is_empty() && c.update_from.is_empty(),
            "{rt:?}"
        );
        assert!(!c.abort && !c.flush_cache && !c.sleep_wake && !c.reports_weight_version);
    }
}

#[test]
fn labels_override_and_mark_source() {
    let c = capabilities_for(
        RuntimeType::Trtllm,
        &labels(&[
            ("rl.pause_modes", "abort, keep"),
            ("rl.update_from", "disk"),
            ("rl.abort", "true"),
            ("rl.flush_cache", "TRUE"),
            ("rl.sleep_wake", "false"),
            ("rl.reports_weight_version", "yes"),
        ]),
    );
    assert_eq!(c.source, RlCapabilitySource::Label);
    assert_eq!(c.pause_modes, ["abort", "keep"]);
    assert_eq!(c.update_from, ["disk"]);
    assert!(c.abort && c.flush_cache && !c.sleep_wake);
    assert!(!c.reports_weight_version, "`yes` is not `true`");
}

#[test]
fn partial_override_keeps_static_rest() {
    let c = capabilities_for(RuntimeType::Sglang, &labels(&[("rl.abort", "false")]));
    assert_eq!(c.source, RlCapabilitySource::Label);
    assert!(!c.abort);
    assert!(c.flush_cache);
    assert_eq!(c.pause_modes, ["abort", "retract", "in_place"]);
}

/// The fallback for TokenSpeed builds that predate advertisement. `keep`
/// is absent on purpose: pre-advertisement builds cannot reach it over
/// HTTP, and an advertising engine overrides this row anyway. `disk` and
/// `tensor` are absent too: the scheduler has no receive path for
/// either, only the trainer-driven NCCL broadcast (`distributed`) works.
#[test]
fn tokenspeed_static_row() {
    let t = capabilities_for(RuntimeType::TokenSpeed, &HashMap::new());
    assert_eq!(t.source, RlCapabilitySource::Static);
    assert_eq!(t.pause_modes, ["wait", "abort"]);
    assert_eq!(t.update_from, ["distributed"]);
    assert!(t.abort && t.flush_cache && t.sleep_wake && t.reports_weight_version);
}

#[test]
fn engine_advertised_labels_override_the_tokenspeed_row() {
    let t = capabilities_for(
        RuntimeType::TokenSpeed,
        &labels(&[
            ("rl.pause_modes", "wait,abort,keep"),
            ("rl.update_from", "distributed"),
        ]),
    );
    assert_eq!(t.source, RlCapabilitySource::Label);
    assert_eq!(t.pause_modes, ["wait", "abort", "keep"]);
    assert_eq!(t.update_from, ["distributed"]);
}
