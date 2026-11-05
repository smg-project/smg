use openai_protocol::worker::{SchedulerLoadSnapshot, WorkerLoadResponse};
use tokio::sync::watch;
use tracing_test::traced_test;

use super::*;
use crate::{
    policies::{BucketConfig, CacheAwareConfig, LeastLoadPolicy, SelectWorkerInfo},
    worker::{
        load_state::LoadSnapshot, test_support::no_health_check, BasicWorkerBuilder, HashRing,
        Worker, WorkerLoadGuard, WorkerType,
    },
};

fn worker(url: &str, worker_type: WorkerType) -> Arc<dyn Worker> {
    Arc::new(
        BasicWorkerBuilder::new(url)
            .worker_type(worker_type)
            .health_config(no_health_check())
            .build(),
    )
}

fn cache_aware_policy() -> Arc<dyn LoadBalancingPolicy> {
    Arc::new(CacheAwarePolicy::with_config(CacheAwareConfig {
        eviction_interval_secs: 0,
        ..Default::default()
    }))
}

fn headers_with_key(key: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("x-smg-routing-key", key.parse().unwrap());
    h
}

#[test]
fn override_eligibility_skips_key_native_policies() {
    // Policies that already honor X-SMG-Routing-Key are skipped; others (incl.
    // prefix_hash, which routes by tokens) get the sticky override.
    assert!(PolicyRegistry::routing_key_override_applies("cache_aware"));
    assert!(PolicyRegistry::routing_key_override_applies("prefix_hash"));
    assert!(PolicyRegistry::routing_key_override_applies("least_load"));
    assert!(!PolicyRegistry::routing_key_override_applies("manual"));
    assert!(!PolicyRegistry::routing_key_override_applies(
        "consistent_hashing"
    ));
}

#[test]
fn override_routes_keyed_request_stickily() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        },
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
        worker("http://w3", WorkerType::Regular),
    ];
    let headers = headers_with_key("session-A");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let first = reg.select_worker(&policy, &workers, &info).unwrap();
    for _ in 0..5 {
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(first));
    }
}

#[test]
fn routing_key_override_keeps_model_pins_independent() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            assignment_mode: ManualAssignmentMode::Delegate,
            ..Default::default()
        },
    );
    let policy = reg.get_default_policy();
    let pools = [
        vec![
            worker("http://a1", WorkerType::Regular),
            worker("http://a2", WorkerType::Regular),
        ],
        vec![worker("http://b1", WorkerType::Regular)],
        vec![worker("http://c1", WorkerType::Regular)],
    ];
    let info = SelectWorkerInfo {
        rid_key: Some("session-42"),
        ..Default::default()
    };

    let first_a = reg
        .select_worker_for_model(&policy, "model-a", &pools[0], &info)
        .unwrap();
    reg.select_worker_for_model(&policy, "model-b", &pools[1], &info)
        .unwrap();
    reg.select_worker_for_model(&policy, "model-c", &pools[2], &info)
        .unwrap();

    assert_eq!(
        reg.select_worker_for_model(&policy, "model-a", &pools[0], &info),
        Some(first_a),
        "other models must not consume this model's bounded failover slots"
    );
}

/// `--routing-key-override` means the same thing under every policy:
/// the routing-key header outranks the body rid. Key-native policies
/// skip the sticky override, so they must apply that precedence themselves.
#[test]
fn header_key_outranks_rid_under_key_native_policies() {
    for config in [
        PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        },
        PolicyConfig::ConsistentHashing,
    ] {
        let reg = PolicyRegistry::with_override(
            config,
            RoutingKeyOverrideConfig {
                enabled: true,
                ..Default::default()
            },
        );
        let policy = reg.get_default_policy();
        let name = policy.name();
        assert!(!PolicyRegistry::routing_key_override_applies(name));
        let workers = vec![
            worker("http://w1", WorkerType::Regular),
            worker("http://w2", WorkerType::Regular),
            worker("http://w3", WorkerType::Regular),
            worker("http://w4", WorkerType::Regular),
        ];
        let hash_ring = Some(Arc::new(HashRing::new(workers.iter().map(|w| w.url()))));
        let headers = headers_with_key("conversation-42");
        let pinned = reg
            .select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    headers: Some(&headers),
                    routing_key: reg.resolve_routing_key(Some(&headers)),
                    hash_ring: hash_ring.clone(),
                    ..Default::default()
                },
            )
            .unwrap();

        for request in 0..4 {
            let rid = format!("request:group:request-{request}");
            let info = SelectWorkerInfo {
                headers: Some(&headers),
                routing_key: reg.resolve_routing_key(Some(&headers)),
                rid_key: reg.derive_rid_key(Some(&rid)),
                hash_ring: hash_ring.clone(),
                ..Default::default()
            };
            assert_eq!(
                reg.select_worker(&policy, &workers, &info),
                Some(pinned),
                "{name}: header must outrank body rid {rid}"
            );
        }
        if let Some(manual) = policy.as_any().downcast_ref::<ManualPolicy>() {
            assert_eq!(manual.map_len(), 1);
        }
    }
}

#[test]
fn override_without_key_uses_configured_policy() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        RoutingKeyOverrideConfig {
            enabled: true,
            ..Default::default()
        },
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let info = SelectWorkerInfo::default(); // no key header
                                            // RoundRobin alternates -> proves the configured policy is used, not sticky.
    let a = reg.select_worker(&policy, &workers, &info).unwrap();
    let b = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_ne!(a, b);
}

fn cache_aware_config() -> PolicyConfig {
    PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 4096)
}

fn headers_with_tokens(value: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert("x-smg-routing-tokens", value.parse().unwrap());
    h
}

fn enabled_override() -> RoutingKeyOverrideConfig {
    RoutingKeyOverrideConfig {
        enabled: true,
        ..Default::default()
    }
}

#[test]
fn request_text_scan_is_false_for_text_free_policies() {
    let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
    assert!(!reg.any_policy_needs_request_text(None));
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with_key("k"))));
}

#[test]
fn tokens_hint_lifts_text_requirement_only_when_valid() {
    let reg = PolicyRegistry::new(cache_aware_config());
    assert!(reg.any_policy_needs_request_text(None));
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
    // Malformed and over-cap hints are ignored, exactly like selection.
    assert!(reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,,3"))));
    let over_cap = vec!["7"; 513].join(",");
    assert!(reg.any_policy_needs_request_text(Some(&headers_with_tokens(&over_cap))));
}

#[test]
fn key_hint_lifts_text_requirement_only_under_enabled_override() {
    let without = PolicyRegistry::new(cache_aware_config());
    assert!(without.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));

    let with = PolicyRegistry::with_override(cache_aware_config(), enabled_override());
    assert!(with.any_policy_needs_request_text(None));
    assert!(!with.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));
    let over_cap = "k".repeat(129);
    assert!(with.any_policy_needs_request_text(Some(&headers_with_key(&over_cap))));
}

#[test]
fn per_model_text_policy_scanned_with_the_same_hint_rules() {
    let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
    assert!(!reg.any_policy_needs_request_text(None));
    reg.on_worker_added("llama-3", Some("cache_aware"));
    assert!(reg.any_policy_needs_request_text(None));
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with_key("session-A"))));
}

/// The answer follows the per-model map: a text policy's model lifts the
/// requirement while one of its workers stands, its last worker's removal
/// drops it, and so does clearing the registry. A model on the text-free
/// default policy changes nothing either way.
#[test]
fn text_requirement_follows_the_per_model_policies() {
    let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
    reg.on_worker_added("plain", None);
    assert!(!reg.any_policy_needs_request_text(None));

    reg.on_worker_added("affine", Some("cache_aware"));
    reg.on_worker_added("affine", Some("cache_aware"));
    assert!(reg.any_policy_needs_request_text(None));
    reg.on_worker_removed("affine");
    assert!(reg.any_policy_needs_request_text(None));
    reg.on_worker_removed("affine");
    assert!(!reg.any_policy_needs_request_text(None));
    reg.on_worker_removed("plain");
    assert!(!reg.any_policy_needs_request_text(None));

    reg.on_worker_added("affine", Some("cache_aware"));
    assert!(reg.any_policy_needs_request_text(None));
    reg.clear();
    assert!(!reg.any_policy_needs_request_text(None));
}

/// The override's exemption is per policy: a sticky-exempt text policy
/// (`consistent_hashing` honours the key itself and reads no text, so the
/// stand-in here is a test policy by that name) keeps the requirement
/// under a valid key where `cache_aware` loses it, as a per-model policy
/// beside the default; the requirement leaves with it. (No hint names
/// the stand-in, so it goes into the map directly and the test refreshes
/// what registration and removal refresh.)
#[test]
fn text_requirement_under_a_key_counts_the_exempt_policies_only() {
    let reg = PolicyRegistry::with_override(cache_aware_config(), enabled_override());
    let keyed = headers_with_key("session-A");
    assert!(reg.any_policy_needs_request_text(None));
    assert!(!reg.any_policy_needs_request_text(Some(&keyed)));
    assert!(!reg.any_policy_needs_request_text_with_tokens(None));
    assert_eq!(
        TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
        TextNeeds(TextNeeds::TEXT)
    );

    reg.model_policies
        .insert("keyed-text".to_string(), Arc::new(TextBesideIdsExempt));
    reg.refresh_text_needs();
    assert!(reg.any_policy_needs_request_text(Some(&keyed)));
    assert!(reg.any_policy_needs_request_text_with_tokens(Some(&keyed)));
    assert!(reg.any_policy_needs_request_text_with_tokens(None));
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with_tokens("1,2,3"))));
    assert_eq!(
        TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
        TextNeeds(
            TextNeeds::TEXT
                | TextNeeds::TEXT_UNDER_KEY
                | TextNeeds::TEXT_WITH_TOKENS
                | TextNeeds::TEXT_WITH_TOKENS_UNDER_KEY
        )
    );

    reg.model_policies.remove("keyed-text");
    reg.refresh_text_needs();
    assert!(!reg.any_policy_needs_request_text(Some(&keyed)));
    assert!(!reg.any_policy_needs_request_text_with_tokens(None));
    assert_eq!(
        TextNeeds(reg.text_needs.load(Ordering::Relaxed)),
        TextNeeds(TextNeeds::TEXT)
    );
}

/// A text policy the routing-key override leaves alone (by name).
#[derive(Debug)]
struct TextBesideIdsExempt;

impl LoadBalancingPolicy for TextBesideIdsExempt {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        (!workers.is_empty()).then_some(0)
    }

    fn name(&self) -> &'static str {
        "consistent_hashing"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A policy that reads the text beside the ids; none of the built-in
/// ones does, so the trait's default is what such a policy gets.
#[derive(Debug)]
struct TextBesideIds;

impl LoadBalancingPolicy for TextBesideIds {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        (!workers.is_empty()).then_some(0)
    }

    fn name(&self) -> &'static str {
        "text_beside_ids"
    }

    fn needs_request_text(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Beside token ids the text is wanted only by a policy that reads it
/// there: `cache_aware` and `bucket` route on the ids and answer no, as
/// the default and as a per-model policy, with the same hint rules; a
/// policy that says nothing about the ids keeps its text answer.
#[test]
fn text_beside_token_ids_is_wanted_only_by_a_policy_that_reads_it_there() {
    let reg = PolicyRegistry::new(cache_aware_config());
    assert!(reg.any_policy_needs_request_text(None));
    assert!(!reg.any_policy_needs_request_text_with_tokens(None));
    assert!(!reg.any_policy_needs_request_text_with_tokens(Some(&headers_with_tokens("1,2,3"))));
    for policy in ["cache_aware", "bucket"] {
        let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
        reg.on_worker_added("llama-3", Some(policy));
        assert!(reg.any_policy_needs_request_text(None), "{policy}");
        assert!(
            !reg.any_policy_needs_request_text_with_tokens(None),
            "{policy}"
        );
    }
    // A per-model policy that reads text beside the ids makes the
    // registry ask for it, under the same hint rule. (The policy goes
    // into the map directly, as no hint names it, so the test refreshes
    // what `on_worker_added` refreshes after its insert.)
    let reg = PolicyRegistry::new(PolicyConfig::RoundRobin);
    reg.model_policies
        .insert("llama-text".to_string(), Arc::new(TextBesideIds));
    reg.refresh_text_needs();
    assert!(reg.any_policy_needs_request_text_with_tokens(None));
    assert!(!reg.any_policy_needs_request_text_with_tokens(Some(&headers_with_tokens("1,2,3"))));
}

#[test]
fn override_disabled_ignores_key() {
    let reg = PolicyRegistry::new(PolicyConfig::RoundRobin); // override off
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let headers = headers_with_key("session-A");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    // Override off -> the key is ignored, RoundRobin alternates.
    let a = reg.select_worker(&policy, &workers, &info).unwrap();
    let b = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_ne!(a, b);
}

fn override_with_headers(names: &[&str]) -> RoutingKeyOverrideConfig {
    RoutingKeyOverrideConfig {
        enabled: true,
        headers: names.iter().map(|n| n.to_string()).collect(),
        ..Default::default()
    }
}

fn headers_with(name: &str, key: &str) -> HeaderMap {
    let mut h = HeaderMap::new();
    h.insert(HeaderName::try_from(name).unwrap(), key.parse().unwrap());
    h
}

#[test]
fn resolve_routing_key_first_present_and_valid_wins() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
    );

    let mut both = headers_with("x-routing-key", "primary");
    both.insert("x-smg-routing-key", "legacy".parse().unwrap());
    assert_eq!(reg.resolve_routing_key(Some(&both)), Some("primary"));

    let legacy_only = headers_with_key("legacy");
    assert_eq!(reg.resolve_routing_key(Some(&legacy_only)), Some("legacy"));

    // An invalid value under the first name falls through to the next.
    let mut over_cap_first = headers_with("x-routing-key", &"k".repeat(129));
    over_cap_first.insert("x-smg-routing-key", "legacy".parse().unwrap());
    assert_eq!(
        reg.resolve_routing_key(Some(&over_cap_first)),
        Some("legacy")
    );

    assert_eq!(reg.resolve_routing_key(None), None);
}

#[test]
fn default_headers_ignore_unconfigured_names() {
    let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
    assert_eq!(
        reg.resolve_routing_key(Some(&headers_with("x-routing-key", "primary"))),
        None
    );
    assert_eq!(
        reg.resolve_routing_key(Some(&headers_with_key("legacy"))),
        Some("legacy")
    );
}

#[test]
fn invalid_configured_header_names_are_skipped() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        override_with_headers(&["not a header", "x-routing-key"]),
    );
    assert_eq!(
        reg.resolve_routing_key(Some(&headers_with("x-routing-key", "primary"))),
        Some("primary")
    );
}

#[test]
fn header_keys_share_rid_lineage_stripping() {
    let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
    assert_eq!(
        reg.sticky_header_key(Some(&headers_with_key("conv_t2"))),
        Some("conv")
    );
    assert_eq!(
        reg.sticky_header_key(Some(&headers_with_key("conv_t2_r1"))),
        Some("conv")
    );
    // A key that is nothing but suffix keys as itself.
    assert_eq!(
        reg.sticky_header_key(Some(&headers_with_key("_t1"))),
        Some("_t1")
    );

    // Override disabled: raw key, no stripping.
    let disabled = PolicyRegistry::new(PolicyConfig::RoundRobin);
    assert_eq!(
        disabled.sticky_header_key(Some(&headers_with_key("conv_t2"))),
        Some("conv_t2")
    );
}

#[test]
fn header_turns_pin_the_same_sticky_entry() {
    let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
        worker("http://w3", WorkerType::Regular),
    ];
    let t1 = headers_with_key("conv_t1");
    let info_t1 = SelectWorkerInfo {
        headers: Some(&t1),
        ..Default::default()
    };
    let first = reg.select_worker(&policy, &workers, &info_t1).unwrap();
    let t2 = headers_with_key("conv_t2");
    let info_t2 = SelectWorkerInfo {
        headers: Some(&t2),
        ..Default::default()
    };
    for _ in 0..5 {
        assert_eq!(reg.select_worker(&policy, &workers, &info_t2), Some(first));
    }
}

#[test]
fn alternate_header_routes_stickily() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
        worker("http://w3", WorkerType::Regular),
    ];
    let headers = headers_with("x-routing-key", "session-A");
    let info = SelectWorkerInfo {
        headers: Some(&headers),
        ..Default::default()
    };
    let first = reg.select_worker(&policy, &workers, &info).unwrap();
    for _ in 0..5 {
        assert_eq!(reg.select_worker(&policy, &workers, &info), Some(first));
    }
}

#[test]
fn alternate_header_lifts_text_gate() {
    let reg = PolicyRegistry::with_override(
        cache_aware_config(),
        override_with_headers(&["x-routing-key", "x-smg-routing-key"]),
    );
    assert!(!reg.any_policy_needs_request_text(Some(&headers_with("x-routing-key", "session-A"))));
    // A name outside the configured list lifts nothing.
    assert!(reg.any_policy_needs_request_text(Some(&headers_with("x-other-key", "session-A"))));
}

fn rid_override(assignment_mode: ManualAssignmentMode) -> RoutingKeyOverrideConfig {
    RoutingKeyOverrideConfig {
        enabled: true,
        assignment_mode,
        ..Default::default()
    }
}

/// A round-robin registry whose routing-key override delegates the assignment, its
/// default policy and the two regular workers the sticky-selection tests route over.
fn delegating_registry_with_two_workers() -> (
    PolicyRegistry,
    Arc<dyn LoadBalancingPolicy>,
    Vec<Arc<dyn Worker>>,
) {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        rid_override(ManualAssignmentMode::Delegate),
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    (reg, policy, workers)
}

#[test]
fn derive_rid_key_strips_turn_and_retry_suffixes() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        rid_override(ManualAssignmentMode::Delegate),
    );
    assert_eq!(reg.derive_rid_key(Some("conv123_t1")), Some("conv123"));
    assert_eq!(reg.derive_rid_key(Some("conv123_t12")), Some("conv123"));
    assert_eq!(reg.derive_rid_key(Some("conv123_t2_r1")), Some("conv123"));
    assert_eq!(reg.derive_rid_key(Some("conv123_r1")), Some("conv123"));
    assert_eq!(reg.derive_rid_key(Some("conv_t1_t2")), Some("conv_t1"));
    assert_eq!(
        reg.derive_rid_key(Some("under_scored_id")),
        Some("under_scored_id")
    );
    assert_eq!(reg.derive_rid_key(Some("no-suffix")), Some("no-suffix"));
    assert_eq!(reg.derive_rid_key(Some("conv_tx1")), Some("conv_tx1"));
    assert_eq!(reg.derive_rid_key(Some("conv_t")), Some("conv_t"));
    // A rid that is nothing but suffix keys as itself.
    assert_eq!(reg.derive_rid_key(Some("_t1")), Some("_t1"));
    assert_eq!(reg.derive_rid_key(None), None);
    let over_cap = format!("{}_t1", "k".repeat(200));
    assert_eq!(reg.derive_rid_key(Some(&over_cap)), None);

    // Override disabled: rid never yields a key.
    let disabled = PolicyRegistry::new(PolicyConfig::RoundRobin);
    assert_eq!(disabled.derive_rid_key(Some("conv123_t1")), None);
}

#[test]
fn header_key_wins_over_rid_and_rid_is_fallback() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();

    let headers = headers_with_key("conversation-42");
    let first = reg
        .select_worker(
            &policy,
            &workers,
            &SelectWorkerInfo {
                headers: Some(&headers),
                rid_key: Some("request:group:request-1"),
                ..Default::default()
            },
        )
        .unwrap();
    for rid in ["request:group:request-2", "request:group:request-3"] {
        assert_eq!(
            reg.select_worker(
                &policy,
                &workers,
                &SelectWorkerInfo {
                    headers: Some(&headers),
                    rid_key: Some(rid),
                    ..Default::default()
                },
            ),
            Some(first)
        );
    }

    assert_eq!(reg.routing_key_sticky.as_ref().unwrap().map_len(), 1);

    // No header: request-ID lineage still gets its own stable pin.
    let rid_info = SelectWorkerInfo {
        rid_key: reg.derive_rid_key(Some("conv42_t1")),
        ..Default::default()
    };
    let pinned = reg.select_worker(&policy, &workers, &rid_info).unwrap();
    assert_ne!(pinned, first);
    assert_eq!(
        reg.select_worker(
            &policy,
            &workers,
            &SelectWorkerInfo {
                rid_key: reg.derive_rid_key(Some("conv42_t2_r1")),
                ..Default::default()
            },
        ),
        Some(pinned)
    );
}

#[test]
fn invalid_header_keys_fall_back_to_rid_for_selection_and_accounting() {
    let reg = PolicyRegistry::with_override(PolicyConfig::RoundRobin, enabled_override());
    let rid_key = reg.derive_rid_key(Some("conversation_t2_r1"));
    let mut invalid_utf8 = HeaderMap::new();
    invalid_utf8.insert(
        "x-smg-routing-key",
        http::HeaderValue::from_bytes(&[0xff]).unwrap(),
    );
    for headers in [
        HeaderMap::new(),
        headers_with_key(""),
        headers_with_key(&"k".repeat(ROUTING_KEY_HINT_MAX_BYTES + 1)),
        invalid_utf8,
    ] {
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            rid_key,
            ..Default::default()
        };
        assert_eq!(
            reg.effective_sticky_key(&info),
            Some(("conversation", "rid"))
        );
        assert_eq!(
            reg.sticky_key(Some(&headers), rid_key),
            Some("conversation")
        );
    }
}

#[test]
fn header_priority_accounts_inflight_requests_under_the_selected_key() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let headers = headers_with_key("conversation_t1");
    let mut guards = Vec::new();
    for rid in ["request-1", "request-2", "request-3"] {
        let info = SelectWorkerInfo {
            headers: Some(&headers),
            rid_key: Some(rid),
            ..Default::default()
        };
        assert_eq!(
            reg.effective_sticky_key(&info),
            Some(("conversation", "header"))
        );
        let selected = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_eq!(selected, usize::from(guards.len() == STICKY_INFLIGHT_CAP));
        guards.push(WorkerLoadGuard::with_key(
            workers[selected].clone(),
            reg.sticky_key(Some(&headers), info.rid_key),
        ));
        assert_eq!(workers[selected].routing_key_inflight(rid), 0);
    }
    assert_eq!(workers[0].routing_key_inflight("conversation"), 2);
    assert_eq!(workers[1].routing_key_inflight("conversation"), 1);
    drop(guards);
    assert_eq!(workers[0].routing_key_inflight("conversation"), 0);
    assert_eq!(workers[1].routing_key_inflight("conversation"), 0);
}

#[test]
fn delegate_assignment_routes_via_underlying_policy_then_pins() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::cache_aware(0.5, 32, 1.1, 0, 10000),
        rid_override(ManualAssignmentMode::Delegate),
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .unwrap()
        .init_workers(&workers);

    let turn1_text = "conversation opening with a long shared instruction block";
    let seeded = reg
        .select_worker(
            &policy,
            &workers,
            &SelectWorkerInfo {
                request_text: Some(turn1_text),
                ..Default::default()
            },
        )
        .unwrap();

    // First keyed request delegates to the policy: the tree match must
    // send it to the seeded worker, not a random pick.
    let key = reg.derive_rid_key(Some("conv42_t1")).unwrap();
    let keyed_turn1 = SelectWorkerInfo {
        request_text: Some(turn1_text),
        rid_key: Some(key),
        ..Default::default()
    };
    assert_eq!(
        reg.select_worker(&policy, &workers, &keyed_turn1),
        Some(seeded)
    );

    // The follow-up shares no routable text; only the pin can send it
    // back. The policy alone would pick the other worker (its tie-break
    // counts prior selections), so equality proves the pin decided.
    let keyed_turn2 = SelectWorkerInfo {
        request_text: Some("unrelated follow-up tail"),
        rid_key: reg.derive_rid_key(Some("conv42_t2")),
        ..Default::default()
    };
    assert_eq!(
        reg.select_worker(&policy, &workers, &keyed_turn2),
        Some(seeded)
    );
}

#[test]
#[traced_test]
fn sticky_selection_emits_decision_line_per_request() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let info = SelectWorkerInfo {
        rid_key: Some("conv1"),
        ..Default::default()
    };
    reg.select_worker(&policy, &workers, &info).unwrap();
    assert!(logs_contain("Sticky routing decision"));
    assert!(logs_contain("vacant"));
    reg.select_worker(&policy, &workers, &info).unwrap();
    assert!(logs_contain("occupied_hit"));
}

/// An overloaded pin must respill through the existing reassignment path.
///
/// Every router hands `select_worker` a slice already filtered by
/// `is_available()`, so a vetoed worker is simply absent from the
/// candidates — the pin resolves to nothing and takes the same
/// `occupied_miss` respill a stale pin takes. The test mirrors that call
/// shape rather than passing a raw pool no router builds.
#[test]
#[traced_test]
fn overloaded_sticky_pin_respills_through_the_stale_path() {
    let (reg, policy, fleet) = delegating_registry_with_two_workers();
    let available = |fleet: &[Arc<dyn Worker>]| -> Vec<Arc<dyn Worker>> {
        fleet.iter().filter(|w| w.is_available()).cloned().collect()
    };
    let info = SelectWorkerInfo {
        rid_key: Some("convOverload"),
        ..Default::default()
    };

    let workers = available(&fleet);
    let pinned_url = {
        let idx = reg.select_worker(&policy, &workers, &info).unwrap();
        assert_eq!(
            reg.select_worker(&policy, &workers, &info),
            Some(idx),
            "the pin holds while the worker is eligible"
        );
        workers[idx].url().to_string()
    };

    fleet
        .iter()
        .find(|w| w.url() == pinned_url)
        .unwrap()
        .set_overloaded(true);
    let workers = available(&fleet);
    assert_eq!(workers.len(), 1, "the veto removes the pinned worker");
    let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_ne!(
        workers[respilled].url(),
        pinned_url,
        "an overloaded pin must be abandoned"
    );
    assert!(logs_contain("occupied_miss"));

    // Recovery re-admits the worker as a candidate; the pin now follows the
    // respill target, so only eligibility is asserted here.
    fleet
        .iter()
        .find(|w| w.url() == pinned_url)
        .unwrap()
        .set_overloaded(false);
    let workers = available(&fleet);
    assert!(reg.select_worker(&policy, &workers, &info).is_some());
}

/// Every worker vetoed leaves the sticky selector with nothing to pin.
#[test]
#[traced_test]
fn all_overloaded_leaves_sticky_selection_empty() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    for w in &workers {
        w.set_overloaded(true);
    }
    let info = SelectWorkerInfo {
        rid_key: Some("convShed"),
        ..Default::default()
    };

    assert_eq!(reg.select_worker(&policy, &workers, &info), None);
    assert!(logs_contain("no_healthy_workers"));
}

#[test]
fn cap_respill_reassigns_and_pin_follows_strict_improvement() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let info = SelectWorkerInfo {
        rid_key: Some("convA"),
        ..Default::default()
    };

    let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(pinned));

    // At the cap (2 of this key in flight on its pin) the key reassigns
    // to the idle worker and the pin follows it, even after the original
    // drains.
    let guards = [
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convA")),
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convA")),
    ];
    let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_ne!(respilled, pinned);
    drop(guards);
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(respilled));
}

#[test]
fn cap_respill_keeps_pin_when_no_strictly_better_pick() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let info = SelectWorkerInfo {
        rid_key: Some("convB"),
        ..Default::default()
    };

    let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
    // Saturate BOTH workers with this key: the reassignment cannot
    // strictly improve (every pick is at the key cap), so the request
    // dispatches wherever the policy says but the pin must survive.
    let mut guards = Vec::new();
    for w in &workers {
        guards.push(WorkerLoadGuard::with_key(w.clone(), Some("convB")));
        guards.push(WorkerLoadGuard::with_key(w.clone(), Some("convB")));
    }
    for _ in 0..4 {
        reg.select_worker(&policy, &workers, &info).unwrap();
    }
    drop(guards);
    assert_eq!(
        reg.select_worker(&policy, &workers, &info),
        Some(pinned),
        "pin must not move to an equally saturated worker"
    );
}

#[test]
fn cap_respill_under_legacy_assignment() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        rid_override(ManualAssignmentMode::MinLoad),
    );
    let policy = reg.get_default_policy();
    let workers = vec![
        worker("http://w1", WorkerType::Regular),
        worker("http://w2", WorkerType::Regular),
    ];
    let info = SelectWorkerInfo {
        rid_key: Some("convC"),
        ..Default::default()
    };

    let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
    let guards = [
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convC")),
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convC")),
    ];
    let respilled = reg.select_worker(&policy, &workers, &info).unwrap();
    assert_ne!(respilled, pinned);
    drop(guards);
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(respilled));
}

#[test]
#[traced_test]
fn busy_worker_does_not_respill_idle_key() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let info = SelectWorkerInfo {
        rid_key: Some("convD"),
        ..Default::default()
    };

    let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
    // Unrelated traffic on the pin, total load far past the cap: the
    // returning key has nothing in flight and must keep its pin.
    let _unrelated = [
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convZ")),
        WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convZ")),
        WorkerLoadGuard::new(workers[pinned].clone(), None),
        WorkerLoadGuard::new(workers[pinned].clone(), None),
    ];
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(pinned));
    assert!(logs_contain("occupied_hit"));
    assert!(!logs_contain("cap_respill"));
}

#[test]
#[traced_test]
fn two_inflight_same_key_respill_the_third() {
    let (reg, policy, workers) = delegating_registry_with_two_workers();
    let info = SelectWorkerInfo {
        rid_key: Some("convE"),
        ..Default::default()
    };

    let pinned = reg.select_worker(&policy, &workers, &info).unwrap();
    let _g1 = WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convE"));
    let _g2 = WorkerLoadGuard::with_key(workers[pinned].clone(), Some("convE"));
    assert_ne!(reg.select_worker(&policy, &workers, &info), Some(pinned));
    assert!(logs_contain("cap_respill"));
}

#[test]
#[traced_test]
fn cap_lifts_when_key_inflight_drains() {
    let reg = PolicyRegistry::with_override(
        PolicyConfig::RoundRobin,
        rid_override(ManualAssignmentMode::Delegate),
    );
    let policy = reg.get_default_policy();
    // Single worker: the respill has nowhere to go, so the branch flips
    // purely on the key's in-flight count.
    let workers = vec![worker("http://w1", WorkerType::Regular)];
    let info = SelectWorkerInfo {
        rid_key: Some("convF"),
        ..Default::default()
    };

    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
    let guards = [
        WorkerLoadGuard::with_key(workers[0].clone(), Some("convF")),
        WorkerLoadGuard::with_key(workers[0].clone(), Some("convF")),
    ];
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
    assert!(logs_contain("cap_respill"));
    drop(guards);
    assert_eq!(reg.select_worker(&policy, &workers, &info), Some(0));
    assert!(logs_contain("occupied_hit"));
}

#[test]
fn test_policy_registry_basic() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // First worker of a model sets the policy
    let policy1 = registry.on_worker_added("llama-3", Some("cache_aware"));
    assert_eq!(policy1.name(), "cache_aware");

    // Second worker of same model uses existing policy
    let policy2 = registry.on_worker_added("llama-3", Some("round_robin"));
    assert_eq!(policy2.name(), "cache_aware"); // Ignores hint, uses existing

    // Different model can have different policy
    let policy3 = registry.on_worker_added("gpt-4", Some("random"));
    assert_eq!(policy3.name(), "random");

    // Check mappings
    let mappings = registry.get_all_mappings();
    assert_eq!(mappings.get("llama-3").unwrap(), "cache_aware");
    assert_eq!(mappings.get("gpt-4").unwrap(), "random");

    // Check worker counts
    let counts = registry.get_worker_counts();
    assert_eq!(*counts.get("llama-3").unwrap(), 2);
    assert_eq!(*counts.get("gpt-4").unwrap(), 1);
}

#[test]
fn test_policy_registry_cleanup() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // Add workers
    registry.on_worker_added("llama-3", Some("cache_aware"));
    registry.on_worker_added("llama-3", None);
    assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&2));

    // Remove one worker - policy should remain
    registry.on_worker_removed("llama-3");
    assert!(registry.get_policy("llama-3").is_some());
    assert_eq!(registry.get_worker_counts().get("llama-3"), Some(&1));

    // Remove last worker - policy should be cleaned up
    registry.on_worker_removed("llama-3");
    assert!(registry.get_policy("llama-3").is_none());
    assert_eq!(registry.get_worker_counts().get("llama-3"), None);
}

/// A policy shared between slots is visited once: a model with no hint
/// takes the registry's default, so one worker's removal reaches that
/// policy once, however many models stand on it, not once per model.
#[test]
fn a_policy_shared_between_slots_is_visited_once() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    for model in ["m1", "m2", "m3"] {
        registry.on_worker_added(model, None);
    }
    registry.on_worker_added("hinted", Some("least_load"));
    let mut visits: Vec<*const ()> = Vec::new();
    registry.each_policy(|policy| visits.push(Arc::as_ptr(policy) as *const ()));
    assert_eq!(
        visits.len(),
        2,
        "the shared default once, the hinted policy once"
    );
    let default = Arc::as_ptr(&registry.default_policy) as *const ();
    assert_eq!(visits.iter().filter(|p| **p == default).count(), 1);
}

#[test]
fn test_passthrough_is_not_load_aware() {
    // Passthrough must not be polled by the WorkerMonitor: with it as the
    // default policy (and a passthrough model policy too), the load-aware
    // set stays empty.
    let registry = PolicyRegistry::new(PolicyConfig::Passthrough);
    registry.on_worker_added("m", Some("passthrough"));
    assert!(!registry.needs_backend_loads());
}

/// The load feed reaches every policy that scores on it, not only the
/// cache-aware ones: a power-of-two default routes on the snapshot the
/// registry handed it, and a model policy created afterwards takes the
/// feed from the registry.
#[test]
fn the_load_feed_reaches_every_policy_that_scores_on_it() {
    fn report(queued_tokens: i32) -> WorkerLoadResponse {
        WorkerLoadResponse {
            loads: vec![SchedulerLoadSnapshot {
                num_waiting_uncached_tokens: queued_tokens,
                gen_throughput: 100.0,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    let registry = PolicyRegistry::new(PolicyConfig::PowerOfTwo {
        load_metric: Default::default(),
    });
    let (_tx, rx) = watch::channel(LoadSnapshot::from_loads_for_test(vec![
        ("http://a:8000".to_string(), report(100_000)),
        ("http://b:8000".to_string(), report(0)),
    ]));
    registry.set_load_receiver(Some(rx));
    let later = registry.on_worker_added("m", Some("least_load"));

    let workers = vec![
        worker("http://a:8000", WorkerType::Regular),
        worker("http://b:8000", WorkerType::Regular),
    ];
    for policy in [registry.get_default_policy(), later] {
        for _ in 0..20 {
            assert_eq!(
                policy.select_worker(&workers, &SelectWorkerInfo::default()),
                Some(1),
                "{} did not route on the feed",
                policy.name()
            );
        }
    }
}

#[test]
fn cache_aware_is_always_load_aware_for_expected_wait() {
    fn cache_aware(
        overlap_decay: f32,
        balance_token_usage_threshold: f32,
        overload_token_usage_threshold: f32,
    ) -> PolicyConfig {
        PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.1,
            eviction_interval_secs: 0,
            max_tree_size: 4096,
            block_size: 16,
            balance_token_usage_threshold,
            overload_token_usage_threshold,
            overlap_decay,
            selection_temperature: 0.0,
            cache_index: Default::default(),
            cache_ttl_secs: 180,
            cache_boundaries: Vec::new(),
            selection_policy: None,
            selection_accounting_ttl_ms: 0,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
        }
    }

    // Even without pressure tuning, misses, spills, and equal-affinity
    // ties use LeastLoad's expected-wait snapshots.
    let plain = PolicyRegistry::new(cache_aware(0.0, 1.0, 1.0));
    assert!(plain.needs_backend_loads());

    // Pressure knobs do not change that load-aware membership.
    for pressured in [
        cache_aware(1.0, 1.0, 1.0),
        cache_aware(0.0, 0.5, 1.0),
        cache_aware(0.0, 1.0, 0.9),
    ] {
        let registry = PolicyRegistry::new(pressured);
        assert!(registry.needs_backend_loads());
    }
}

#[test]
fn test_default_policy() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);

    // No hint, no template - uses default
    let policy = registry.on_worker_added("unknown-model", None);
    assert_eq!(policy.name(), "round_robin");

    // Get default directly
    let default = registry.get_default_policy();
    assert_eq!(default.name(), "round_robin");
}

#[test]
fn hinted_policy_inherits_operator_cache_aware_config() {
    let registry = PolicyRegistry::new(PolicyConfig::CacheAware {
        cache_threshold: 0.9,
        balance_abs_threshold: 64,
        balance_rel_threshold: 2.5,
        eviction_interval_secs: 0,
        max_tree_size: 4096,
        block_size: 32,
        balance_token_usage_threshold: 0.5,
        overload_token_usage_threshold: 0.8,
        overlap_decay: 0.0,
        selection_temperature: 0.0,
        cache_index: Default::default(),
        cache_ttl_secs: 180,
        cache_boundaries: Vec::new(),
        selection_policy: None,
        selection_accounting_ttl_ms: 0,
        balance_prefix_margin_tokens: 0,
        balance_uncached_credit: false,
    });

    // Hinted policy is a fresh per-model instance, not the shared default.
    let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
    assert!(!Arc::ptr_eq(&policy, &registry.get_default_policy()));

    let config = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .unwrap()
        .config_for_test();
    assert_eq!(config.cache_threshold, 0.9);
    assert_eq!(config.balance_abs_threshold, 64);
    assert_eq!(config.balance_rel_threshold, 2.5);
    assert_eq!(config.eviction_interval_secs, 0);
    assert_eq!(config.max_tree_size, 4096);
    assert_eq!(config.block_size, 32);
    assert_eq!(config.balance_token_usage_threshold, 0.5);
    assert_eq!(config.overload_token_usage_threshold, 0.8);

    // Aliases accepted by create_by_name inherit too.
    let alias = registry.on_worker_added("llama-4", Some("CacheAware"));
    let alias_config = alias
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .unwrap()
        .config_for_test();
    assert_eq!(alias_config.max_tree_size, 4096);
}

#[test]
fn hinted_policy_inherits_operator_least_load_config() {
    let registry = PolicyRegistry::new(PolicyConfig::LeastLoad {
        kv_pressure_weight: 0.4,
        mean_prefill_tokens: 2048,
        default_throughput: 555.0,
    });

    let policy = registry.on_worker_added("llama-3", Some("least_load"));
    assert!(!Arc::ptr_eq(&policy, &registry.get_default_policy()));
    let least_load = policy.as_any().downcast_ref::<LeastLoadPolicy>().unwrap();
    assert_eq!(least_load.params_for_test(), (0.4, 2048, 555.0));
}

/// An integration setter racing policy publication must never be
/// missed: publication injects and inserts under the integration
/// guards, so the setter either wrote first (its value is injected)
/// or its propagation scan runs after the insert.
#[test]
fn hinted_policy_publication_races_integration_setter() {
    use std::sync::Barrier;

    use crate::worker::KvEventMonitor;

    const ADDERS: usize = 4;

    let registry = Arc::new(PolicyRegistry::new(PolicyConfig::cache_aware(
        0.5, 32, 1.5, 0, 128,
    )));

    for round in 0..64 {
        let monitor = Arc::new(KvEventMonitor::new());
        let barrier = Arc::new(Barrier::new(ADDERS + 1));
        let handles: Vec<_> = (0..ADDERS)
            .map(|k| {
                let registry = Arc::clone(&registry);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    registry.on_worker_added(&format!("model-{round}-{k}"), Some("cache_aware"));
                })
            })
            .collect();
        barrier.wait();
        registry.set_kv_event_monitor(Some(monitor));
        for handle in handles {
            handle.join().unwrap();
        }
        for entry in registry.model_policies.iter() {
            let cache_aware = entry
                .value()
                .as_any()
                .downcast_ref::<CacheAwarePolicy>()
                .unwrap();
            assert!(
                cache_aware.kv_event_monitor_is_set_for_test(),
                "policy for {} missed the concurrently-set monitor",
                entry.key()
            );
        }
        registry.set_kv_event_monitor(None);
        registry.clear();
    }
}

/// The PD/EPD policies are set after the app context wired the shared state
/// into the registry (the KV event monitor, the load feed, and the mesh
/// tree bridge that `MeshAdapters::start` attaches), so the policy setters
/// hand all of it over themselves.
#[tokio::test]
async fn pd_role_policies_receive_the_shared_state_set_before_them() {
    use std::collections::BTreeMap;

    use smg_mesh::MeshKV;

    use crate::{mesh::MeshAdapters, worker::WorkerRegistry};

    let registry = Arc::new(PolicyRegistry::new(PolicyConfig::RoundRobin));
    registry.set_kv_event_monitor(Some(Arc::new(KvEventMonitor::new())));
    let (_load_tx, load_rx) = watch::channel(LoadSnapshot::from_loads_for_test(Vec::new()));
    registry.set_load_receiver(Some(load_rx));
    let mesh = MeshKV::new("node-a".into());
    let _adapters = MeshAdapters::start(
        &mesh,
        "node-a".into(),
        Arc::new(WorkerRegistry::new()),
        Arc::new(RwLock::new(BTreeMap::new())),
        Arc::clone(&registry),
    );

    let prefill = cache_aware_policy();
    let decode = cache_aware_policy();
    let encode = cache_aware_policy();
    registry.set_prefill_policy(Arc::clone(&prefill));
    registry.set_decode_policy(Arc::clone(&decode));
    registry.set_encode_policy(Arc::clone(&encode));

    for (role, policy) in [("prefill", prefill), ("decode", decode), ("encode", encode)] {
        let cache_aware = policy.as_any().downcast_ref::<CacheAwarePolicy>().unwrap();
        assert!(
            cache_aware.kv_event_monitor_is_set_for_test(),
            "{role} policy missed the KV event monitor"
        );
        assert!(
            cache_aware.has_load_receiver_for_test(),
            "{role} policy missed the load feed"
        );
        assert!(
            cache_aware.should_populate_hash_index_for_test(),
            "{role} policy missed the mesh tree bridge"
        );
    }
}

#[test]
fn hinted_policy_without_operator_config_uses_type_defaults() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    let policy = registry.on_worker_added("llama-3", Some("cache_aware"));
    let config = policy
        .as_any()
        .downcast_ref::<CacheAwarePolicy>()
        .unwrap()
        .config_for_test();
    let defaults = CacheAwareConfig::default();
    assert_eq!(config.cache_threshold, defaults.cache_threshold);
    assert_eq!(config.max_tree_size, defaults.max_tree_size);
    assert_eq!(
        config.eviction_interval_secs,
        defaults.eviction_interval_secs
    );
}

#[test]
fn test_pd_cache_aware_policy_initialization() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    registry.set_prefill_policy(cache_aware_policy());
    registry.set_decode_policy(cache_aware_policy());

    let prefill_workers = vec![
        worker("http://prefill-1:8000", WorkerType::Prefill),
        worker("http://prefill-2:8000", WorkerType::Prefill),
    ];
    let decode_workers = vec![
        worker("http://decode-1:8000", WorkerType::Decode),
        worker("http://decode-2:8000", WorkerType::Decode),
    ];

    registry.init_pd_cache_aware_policies(&prefill_workers, &decode_workers);

    let prefill_policy = registry.get_prefill_policy();
    let decode_policy = registry.get_decode_policy();
    let info = SelectWorkerInfo {
        request_text: Some("shared prefix request"),
        ..Default::default()
    };

    let prefill_first = prefill_policy.select_worker(&prefill_workers, &info);
    let prefill_second = prefill_policy.select_worker(&prefill_workers, &info);
    assert!(prefill_first.is_some());
    assert_eq!(prefill_first, prefill_second);

    let decode_first = decode_policy.select_worker(&decode_workers, &info);
    let decode_second = decode_policy.select_worker(&decode_workers, &info);
    assert!(decode_first.is_some());
    assert_eq!(decode_first, decode_second);

    registry.remove_worker_from_pd_cache_aware("http://prefill-1:8000");
    registry.remove_worker_from_pd_cache_aware("http://decode-1:8000");
}

/// A policy that remembers completions and reconciliation ticks.
#[derive(Debug, Default)]
struct CompletionRecorder {
    completed: std::sync::Mutex<Vec<String>>,
    reconciled: std::sync::Mutex<Vec<(String, usize)>>,
}

impl LoadBalancingPolicy for CompletionRecorder {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        (!workers.is_empty()).then_some(0)
    }

    fn on_request_complete(&self, worker_url: &str, _success: bool) {
        self.completed.lock().unwrap().push(worker_url.to_string());
    }

    fn reconcile_in_flight(&self, worker_url: &str, in_flight: usize) {
        self.reconciled
            .lock()
            .unwrap()
            .push((worker_url.to_string(), in_flight));
    }

    fn name(&self) -> &'static str {
        "completion_recorder"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[test]
fn completion_sink_reports_to_the_policy_that_owns_the_worker_type() {
    use crate::worker::{BasicWorkerBuilder, WorkerLoadGuard};

    let registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
    let prefill_policy = Arc::new(CompletionRecorder::default());
    let decode_policy = Arc::new(CompletionRecorder::default());
    registry.set_prefill_policy(Arc::clone(&prefill_policy) as Arc<dyn LoadBalancingPolicy>);
    registry.set_decode_policy(Arc::clone(&decode_policy) as Arc<dyn LoadBalancingPolicy>);
    let sink = registry.completion_sink();

    let prefill: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://prefill:9000")
            .worker_type(WorkerType::Prefill)
            .build(),
    );
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:9000")
            .worker_type(WorkerType::Decode)
            .build(),
    );
    let regular: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://regular:9000")
            .worker_type(WorkerType::Regular)
            .build(),
    );
    for worker in [&prefill, &decode, &regular] {
        worker.set_completion_sink(Some(Arc::clone(&sink)));
    }

    drop(WorkerLoadGuard::new(Arc::clone(&prefill), None));
    drop(WorkerLoadGuard::new(Arc::clone(&decode), None));
    // The regular worker's model policy (random) accepts the completion silently.
    drop(WorkerLoadGuard::new(Arc::clone(&regular), None));

    assert_eq!(
        prefill_policy.completed.lock().unwrap().as_slice(),
        ["grpc://prefill:9000"]
    );
    assert_eq!(
        decode_policy.completed.lock().unwrap().as_slice(),
        ["grpc://decode:9000"]
    );

    // A worker that outlives its registry reports to nobody.
    drop(registry);
    drop(WorkerLoadGuard::new(Arc::clone(&prefill), None));
    assert_eq!(prefill_policy.completed.lock().unwrap().len(), 1);
}

#[test]
fn reconcile_in_flight_reaches_the_policy_that_owns_the_worker_type() {
    use crate::worker::{BasicWorkerBuilder, WorkerLoadGuard};

    let registry = Arc::new(PolicyRegistry::new(PolicyConfig::Random));
    let decode_policy = Arc::new(CompletionRecorder::default());
    registry.set_decode_policy(Arc::clone(&decode_policy) as Arc<dyn LoadBalancingPolicy>);
    let decode: Arc<dyn Worker> = Arc::new(
        BasicWorkerBuilder::new("grpc://decode:9000")
            .worker_type(WorkerType::Decode)
            .build(),
    );

    // The count handed over is the router's live one: two guards held,
    // then none.
    let held = [
        WorkerLoadGuard::new(Arc::clone(&decode), None),
        WorkerLoadGuard::new(Arc::clone(&decode), None),
    ];
    registry.reconcile_in_flight(decode.as_ref());
    drop(held);
    registry.reconcile_in_flight(decode.as_ref());
    assert_eq!(
        decode_policy.reconciled.lock().unwrap().as_slice(),
        [
            ("grpc://decode:9000".to_string(), 2),
            ("grpc://decode:9000".to_string(), 0)
        ]
    );
}

/// A policy that remembers which workers it was told to forget and
/// consumes no load reports.
#[derive(Debug, Default)]
struct RemovalRecorder {
    removed: std::sync::Mutex<Vec<String>>,
}

impl LoadBalancingPolicy for RemovalRecorder {
    fn select_worker(
        &self,
        workers: &[Arc<dyn Worker>],
        _info: &SelectWorkerInfo,
    ) -> Option<usize> {
        (!workers.is_empty()).then_some(0)
    }

    fn remove_worker(&self, url: &str) {
        self.removed.lock().unwrap().push(url.to_string());
    }

    fn name(&self) -> &'static str {
        "removal_recorder"
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A worker's removal reaches every policy instance once: the ones that
/// consume load reports and the ones that do not, a shared instance once
/// however many roles it serves.
#[test]
fn worker_removal_reaches_every_policy_once() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    let shared = Arc::new(RemovalRecorder::default());
    registry.set_prefill_policy(Arc::clone(&shared) as Arc<dyn LoadBalancingPolicy>);
    registry.set_decode_policy(Arc::clone(&shared) as Arc<dyn LoadBalancingPolicy>);
    let per_model = Arc::new(RemovalRecorder::default());
    registry.model_policies.insert(
        "m1".to_string(),
        Arc::clone(&per_model) as Arc<dyn LoadBalancingPolicy>,
    );

    registry.remove_worker_from_policies("http://w2:8000");

    assert_eq!(
        shared.removed.lock().unwrap().as_slice(),
        ["http://w2:8000"]
    );
    assert_eq!(
        per_model.removed.lock().unwrap().as_slice(),
        ["http://w2:8000"]
    );
}

/// A prefill worker's removal reaches the bucket policy, which keeps the
/// prefill workers' size partition without consuming load reports: every
/// request size still selects one of the workers left.
#[test]
fn removing_a_prefill_worker_reaches_the_bucket_policy() {
    let registry = PolicyRegistry::new(PolicyConfig::RoundRobin);
    registry.set_prefill_policy(Arc::new(BucketPolicy::with_config(BucketConfig {
        balance_abs_threshold: usize::MAX,
        bucket_adjust_interval_secs: 3600,
        ..Default::default()
    })));
    let prefill = vec![
        worker("http://prefill-1:8000", WorkerType::Prefill),
        worker("http://prefill-2:8000", WorkerType::Prefill),
        worker("http://prefill-3:8000", WorkerType::Prefill),
    ];
    registry.init_pd_bucket_policies(&prefill);

    registry.remove_worker_from_policies("http://prefill-2:8000");

    let remaining = [Arc::clone(&prefill[0]), Arc::clone(&prefill[2])];
    let policy = registry.get_prefill_policy();
    for size in [0, 1364, 1365, 2000, 2729, 2730, 10_000] {
        let text = "a".repeat(size);
        let info = SelectWorkerInfo {
            request_text: Some(&text),
            ..Default::default()
        };
        let selected = policy
            .select_worker(&remaining, &info)
            .unwrap_or_else(|| panic!("a request of {size} chars selects no prefill worker"));
        assert_ne!(
            remaining[selected].url(),
            "http://prefill-2:8000",
            "size {size}"
        );
    }
}
