//! The policy document the gateway gives to its controller.
//!
//! The policy crate converts routing settings into the controller's
//! document: [`RoutingSettings`] says where a setting lands, what is refused
//! and what is left out. This module maps the gateway's configuration onto
//! that type, hands the gateway's own defaults to the comparison with a
//! supplied document, and keeps the check that reads the gateway's tenant
//! header. No request path calls this module.
//!
//! The routing settings are described twice: by this crate's configuration
//! types, which the flags, the configuration file and the bindings fill, and
//! by the policy crate's input type. The map below ties the two. It takes
//! each policy, each mode and the sticky override apart field by field, with
//! no `..`, so a setting added to one of them does not build until the map
//! carries it.

use smg_policy_controller::document::{
    AppliedSpec, Assignment, IndexKind, LoadMetricSetting, ModeSettings, Pairing, PolicySettings,
    PolicySpec, RoutingKeyOverrideSettings, RoutingSettings, RoutingSettingsError, SpecError,
};

use super::{
    CacheIndexKind, GroupedDpSticky, ManualAssignmentMode, PdPairingMode, PolicyConfig,
    PowerOfTwoLoadMetric, RouterConfig, RoutingKeyOverrideConfig, RoutingMode,
};
use crate::policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY;

impl RouterConfig {
    /// The policy document this configuration says, or the first of its
    /// settings that no document says.
    ///
    /// A pure function of the configuration. The document is not validated
    /// here: the controller validates a document where it is applied, and
    /// names there a stage its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, RoutingSettingsError> {
        self.routing_settings().policy_document()
    }

    /// The document this gateway runs. Without a supplied document, it is
    /// the one the routing settings mean. A supplied document takes effect
    /// if the settings describe the same document or the default
    /// configuration's document for this mode. A conflict names the first
    /// difference and the two values.
    ///
    /// Both launchers refuse a routing setting beside a document before
    /// this runs, so the comparison judges a configuration built by hand.
    pub fn policy_document_in_force(&self) -> Result<PolicySpec, RoutingSettingsError> {
        let defaults = Self::default().routing_settings();
        self.routing_settings().policy_document_in_force(&defaults)
    }

    /// Validates the document over the stock stages, with the gateway's
    /// own headers denied to the view: the typed half of the check a hosted
    /// document passes. The start and `--print-policy-spec` hand the
    /// document's YAML to the operators' door, which bounds its size and
    /// reads it back before it validates it this way.
    pub fn hosted_policy(&self, document: &PolicySpec) -> Result<AppliedSpec, SpecError> {
        AppliedSpec::validate_with(document, &self.denied_view_headers())
    }

    /// The header names this gateway denies the view of a document it
    /// hosts: its tenant header, which it keeps to itself.
    pub(crate) fn denied_view_headers(&self) -> [&str; 1] {
        [self.tenant_resolution.tenant_header_name.as_str()]
    }

    /// This configuration's routing settings in the policy crate's form,
    /// each under its own name. A mode's pools are the gateway's own and
    /// are left behind.
    fn routing_settings(&self) -> RoutingSettings {
        let side = |policy: Option<&PolicyConfig>| policy.map(policy_settings);
        let mode = match &self.mode {
            RoutingMode::Regular { worker_urls: _ } => ModeSettings::Regular,
            RoutingMode::PrefillDecode {
                prefill_urls: _,
                decode_urls: _,
                prefill_policy,
                decode_policy,
            } => ModeSettings::PrefillDecode {
                prefill_policy: side(prefill_policy.as_ref()),
                decode_policy: side(decode_policy.as_ref()),
            },
            RoutingMode::EncodePrefillDecode {
                encode_urls: _,
                prefill_urls: _,
                decode_urls: _,
                encode_policy,
                prefill_policy,
                decode_policy,
            } => ModeSettings::EncodePrefillDecode {
                encode_policy: side(encode_policy.as_ref()),
                prefill_policy: side(prefill_policy.as_ref()),
                decode_policy: side(decode_policy.as_ref()),
            },
            RoutingMode::OpenAI { worker_urls: _ }
            | RoutingMode::Anthropic { worker_urls: _ }
            | RoutingMode::Gemini { worker_urls: _ } => ModeSettings::Provider(self.mode_type()),
        };
        let RoutingKeyOverrideConfig {
            enabled,
            eviction_interval_secs,
            max_idle_secs,
            assignment_mode,
            headers,
        } = &self.routing_key_override;
        RoutingSettings {
            mode,
            policy: policy_settings(&self.policy),
            enable_igw: self.enable_igw,
            policy_spec: self.policy_spec.clone(),
            grouped_dp_sticky: match self.grouped_dp_sticky {
                GroupedDpSticky::Off => None,
                pin @ GroupedDpSticky::Hash => Some(pin.as_str()),
            },
            dp_minimum_tokens_scheduler: self.dp_minimum_tokens_scheduler,
            pd_pairing_mode: match self.pd_pairing_mode {
                PdPairingMode::Off => Pairing::Off,
                PdPairingMode::Lenient => Pairing::Lenient,
                PdPairingMode::Strict => Pairing::Strict,
            },
            worker_warmup_secs: self.worker_warmup_secs,
            worker_warmup_share: self.worker_warmup_share,
            worker_warmup_blocks: self.worker_warmup_blocks,
            worker_warmup_thin_ratio: self.worker_warmup_thin_ratio,
            worker_warmup_divert_every: self.worker_warmup_divert_every,
            routing_key_override: RoutingKeyOverrideSettings {
                enabled: *enabled,
                headers: headers.clone(),
                eviction_interval_secs: *eviction_interval_secs,
                max_idle_secs: *max_idle_secs,
                assignment_mode: assignment(*assignment_mode),
            },
            worker_overload_protection: self.worker_overload_protection,
            worker_overload_waiting_requests: self.worker_overload_waiting_requests,
            worker_overload_token_usage: self.worker_overload_token_usage,
            worker_overload_shed: self.worker_overload_shed,
        }
    }
}

/// How a key seen for the first time picks its worker, in the policy
/// crate's form: the manual policy's setting and the sticky override's.
fn assignment(mode: ManualAssignmentMode) -> Assignment {
    match mode {
        ManualAssignmentMode::Random => Assignment::Random,
        ManualAssignmentMode::MinLoad => Assignment::MinLoad,
        ManualAssignmentMode::MinGroup => Assignment::MinGroup,
        ManualAssignmentMode::Delegate => Assignment::Delegate,
    }
}

/// A policy in the policy crate's form, each setting under its own name.
fn policy_settings(policy: &PolicyConfig) -> PolicySettings {
    match policy.clone() {
        PolicyConfig::Random => PolicySettings::Random,
        PolicyConfig::RoundRobin => PolicySettings::RoundRobin,
        PolicyConfig::Passthrough => PolicySettings::Passthrough,
        PolicyConfig::ConsistentHashing => PolicySettings::ConsistentHashing,
        PolicyConfig::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            balance_prefix_margin_tokens,
            balance_uncached_credit,
            eviction_interval_secs,
            max_tree_size,
            block_size,
            balance_token_usage_threshold,
            overload_token_usage_threshold,
            overlap_decay,
            selection_temperature,
            cache_index,
            cache_ttl_secs,
            cache_boundaries,
            selection_policy,
            selection_accounting_ttl_ms,
        } => PolicySettings::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            balance_prefix_margin_tokens,
            balance_uncached_credit,
            eviction_interval_secs,
            max_tree_size,
            block_size,
            balance_token_usage_threshold,
            overload_token_usage_threshold,
            overlap_decay,
            selection_temperature,
            cache_index: match cache_index {
                CacheIndexKind::Tree => IndexKind::Tree,
                CacheIndexKind::Hash => IndexKind::Hash,
            },
            cache_ttl_secs,
            cache_boundaries,
            // The policy crate takes a selection policy other than the
            // default one, which no document says. The default's own name
            // is the default.
            selection_policy: selection_policy.filter(|named| named != DEFAULT_SELECTION_POLICY),
            selection_accounting_ttl_ms,
        },
        PolicyConfig::PowerOfTwo { load_metric } => PolicySettings::PowerOfTwo {
            load_metric: match load_metric {
                PowerOfTwoLoadMetric::ExpectedWait => LoadMetricSetting::ExpectedWait,
                PowerOfTwoLoadMetric::Requests => LoadMetricSetting::Requests,
                PowerOfTwoLoadMetric::LeastRequests => LoadMetricSetting::LeastRequests,
                PowerOfTwoLoadMetric::ResidentTokens { request_tokens } => {
                    LoadMetricSetting::ResidentTokens { request_tokens }
                }
            },
        },
        PolicyConfig::LeastLoad {
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
        } => PolicySettings::LeastLoad {
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
        },
        PolicyConfig::Bucket {
            balance_abs_threshold,
            balance_rel_threshold,
            bucket_adjust_interval_secs,
        } => PolicySettings::Bucket {
            balance_abs_threshold,
            balance_rel_threshold,
            bucket_adjust_interval_secs,
        },
        PolicyConfig::Manual {
            eviction_interval_secs,
            max_idle_secs,
            assignment_mode,
        } => PolicySettings::Manual {
            eviction_interval_secs,
            max_idle_secs,
            assignment_mode: assignment(assignment_mode),
        },
        PolicyConfig::PrefixHash {
            prefix_token_count,
            load_factor,
            balance_abs_threshold,
            cache_boundaries,
        } => PolicySettings::PrefixHash {
            prefix_token_count,
            load_factor,
            balance_abs_threshold,
            cache_boundaries,
        },
    }
}

#[cfg(test)]
mod tests {
    use smg_policy_controller::document::PRESETS;

    use super::*;

    fn config(policy: PolicyConfig) -> RouterConfig {
        RouterConfig {
            policy,
            ..RouterConfig::default()
        }
    }

    /// The policy of a preset's name with every setting at its flag's
    /// default.
    fn at_flag_defaults(name: &str) -> PolicyConfig {
        match name {
            "random" => PolicyConfig::Random,
            "round_robin" => PolicyConfig::RoundRobin,
            "passthrough" => PolicyConfig::Passthrough,
            "consistent_hashing" => PolicyConfig::ConsistentHashing,
            "cache_aware" => PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864),
            "power_of_two" => PolicyConfig::PowerOfTwo {
                load_metric: PowerOfTwoLoadMetric::ExpectedWait,
            },
            "least_load" => PolicyConfig::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
            "bucket" => PolicyConfig::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            },
            "manual" => PolicyConfig::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: ManualAssignmentMode::Random,
            },
            "prefix_hash" => PolicyConfig::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            },
            other => {
                panic!("the controller has a preset this module converts no policy to: {other}")
            }
        }
    }

    fn preset(name: &str) -> PolicySpec {
        PolicySpec::preset(name).expect("a preset of this build")
    }

    /// The gateway's own defaults are the controller's presets: the default
    /// configuration, each policy at its flags' defaults, and the map
    /// between them.
    #[test]
    fn each_policy_name_at_its_flags_defaults_is_its_preset() {
        for name in PRESETS {
            let policy = at_flag_defaults(name);
            assert_eq!(
                policy.name(),
                name,
                "the policy and the preset share one name"
            );
            assert_eq!(
                config(policy).policy_document(),
                Ok(preset(name)),
                "{name} at its flags' defaults is the preset, with nothing written over it"
            );
        }
    }

    /// The comparison with a supplied document reads the gateway's own
    /// default configuration.
    #[test]
    fn a_given_document_wins_where_no_routing_setting_is_set() {
        for (mode, document) in [
            (RouterConfig::default().mode, preset("round_robin")),
            (
                RoutingMode::PrefillDecode {
                    prefill_urls: vec![],
                    decode_urls: vec![],
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicySpec::pair_presets("round_robin", "random").expect("paired presets"),
            ),
        ] {
            let config = RouterConfig {
                mode,
                policy_spec: Some(document.clone()),
                ..RouterConfig::default()
            };
            assert_eq!(config.policy_document_in_force(), Ok(document));
        }
    }

    #[test]
    fn the_pin_to_an_engine_of_a_grouped_worker_moves_no_document() {
        // The dispatch pins a sticky key behind the decision, on the worker
        // the decision names: the document is the same with the pin on.
        let pinned = RouterConfig {
            grouped_dp_sticky: GroupedDpSticky::Hash,
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(pinned.policy_document(), Ok(preset("round_robin")));
        assert_eq!(pinned.policy_document_in_force(), Ok(preset("round_robin")));
        // The pin reads the gateway's own key, and a supplied document names
        // its own key sources: beside one the pin is refused by its name.
        let given = RouterConfig {
            grouped_dp_sticky: GroupedDpSticky::Hash,
            policy_spec: Some(preset("least_load")),
            ..RouterConfig::default()
        };
        assert_eq!(
            given.policy_document_in_force(),
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "grouped_dp_sticky",
                value: "hash".to_owned(),
                reason: "beside a supplied document the pin would read the gateway's own key, which that document's key sources need not be",
            })
        );
    }

    /// Each setting holds a value of its own, so a setting the map drops or
    /// writes under another name shows.
    #[test]
    fn every_routing_setting_reaches_the_policy_crate_under_its_name() {
        let document = preset("round_robin");
        let mut config = RouterConfig {
            mode: RoutingMode::Regular {
                worker_urls: vec!["http://worker:8000".to_owned()],
            },
            policy: PolicyConfig::RoundRobin,
            enable_igw: true,
            policy_spec: Some(document.clone()),
            grouped_dp_sticky: GroupedDpSticky::Hash,
            dp_minimum_tokens_scheduler: true,
            pd_pairing_mode: PdPairingMode::Strict,
            worker_warmup_secs: 7,
            worker_warmup_share: 0.75,
            worker_warmup_blocks: 9,
            worker_warmup_thin_ratio: 0.125,
            worker_warmup_divert_every: 11,
            worker_overload_protection: false,
            worker_overload_waiting_requests: Some(13),
            worker_overload_token_usage: Some(0.625),
            worker_overload_shed: true,
            ..RouterConfig::default()
        };
        config.routing_key_override.enabled = true;
        config.routing_key_override.headers = vec!["x-session".to_owned()];
        config.routing_key_override.eviction_interval_secs = 15;
        config.routing_key_override.max_idle_secs = 17;
        config.routing_key_override.assignment_mode = ManualAssignmentMode::MinGroup;
        assert_eq!(
            config.routing_settings(),
            RoutingSettings {
                mode: ModeSettings::Regular,
                policy: PolicySettings::RoundRobin,
                enable_igw: true,
                policy_spec: Some(document),
                grouped_dp_sticky: Some("hash"),
                dp_minimum_tokens_scheduler: true,
                pd_pairing_mode: Pairing::Strict,
                worker_warmup_secs: 7,
                worker_warmup_share: 0.75,
                worker_warmup_blocks: 9,
                worker_warmup_thin_ratio: 0.125,
                worker_warmup_divert_every: 11,
                routing_key_override: RoutingKeyOverrideSettings {
                    enabled: true,
                    headers: vec!["x-session".to_owned()],
                    eviction_interval_secs: 15,
                    max_idle_secs: 17,
                    assignment_mode: Assignment::MinGroup,
                },
                worker_overload_protection: false,
                worker_overload_waiting_requests: Some(13),
                worker_overload_token_usage: Some(0.625),
                worker_overload_shed: true,
            }
        );

        // The pin off is no pin, and each pairing mode is its namesake.
        let plain = RouterConfig::default().routing_settings();
        assert_eq!(plain.grouped_dp_sticky, None);
        for (mode, pairing) in [
            (PdPairingMode::Off, Pairing::Off),
            (PdPairingMode::Lenient, Pairing::Lenient),
            (PdPairingMode::Strict, Pairing::Strict),
        ] {
            let config = RouterConfig {
                pd_pairing_mode: mode,
                ..RouterConfig::default()
            };
            assert_eq!(config.routing_settings().pd_pairing_mode, pairing);
        }
    }

    /// The policy crate's tests of the translation run over two literals of
    /// this gateway's defaults: `defaults()` and `cache_aware()` in its
    /// `document/routing_settings.rs`. The two literals below are those. A
    /// default that moves here fails this case, and the crate's literal
    /// moves with it.
    #[test]
    fn the_gateways_defaults_are_the_policy_crates_fixtures_of_them() {
        assert_eq!(
            RouterConfig::default().routing_settings(),
            RoutingSettings {
                mode: ModeSettings::Regular,
                policy: PolicySettings::Random,
                enable_igw: false,
                policy_spec: None,
                grouped_dp_sticky: None,
                dp_minimum_tokens_scheduler: false,
                pd_pairing_mode: Pairing::Lenient,
                worker_warmup_secs: 60,
                worker_warmup_share: 0.25,
                worker_warmup_blocks: 1024,
                worker_warmup_thin_ratio: 0.5,
                worker_warmup_divert_every: 8,
                routing_key_override: RoutingKeyOverrideSettings {
                    enabled: false,
                    headers: vec!["x-smg-routing-key".to_owned()],
                    eviction_interval_secs: 60,
                    max_idle_secs: 14_400,
                    assignment_mode: Assignment::Delegate,
                },
                worker_overload_protection: true,
                worker_overload_waiting_requests: Some(8),
                worker_overload_token_usage: Some(0.8),
                worker_overload_shed: false,
            }
        );
        assert_eq!(
            policy_settings(&PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864)),
            PolicySettings::CacheAware {
                cache_threshold: 0.3,
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                balance_prefix_margin_tokens: 0,
                balance_uncached_credit: false,
                eviction_interval_secs: 120,
                max_tree_size: 67_108_864,
                block_size: 16,
                balance_token_usage_threshold: 1.0,
                overload_token_usage_threshold: 1.0,
                overlap_decay: 0.0,
                selection_temperature: 0.0,
                cache_index: IndexKind::Tree,
                cache_ttl_secs: 180,
                cache_boundaries: Vec::new(),
                selection_policy: None,
                selection_accounting_ttl_ms: 0,
            }
        );
    }

    #[test]
    fn every_policy_reaches_the_policy_crate_with_its_settings() {
        let cache_aware = PolicyConfig::CacheAware {
            cache_threshold: 0.5,
            balance_abs_threshold: 2,
            balance_rel_threshold: 3.5,
            balance_prefix_margin_tokens: 4,
            balance_uncached_credit: true,
            eviction_interval_secs: 5,
            max_tree_size: 6,
            block_size: 7,
            balance_token_usage_threshold: 0.25,
            overload_token_usage_threshold: 0.75,
            overlap_decay: 0.125,
            selection_temperature: 0.0625,
            cache_index: CacheIndexKind::Hash,
            cache_ttl_secs: 8,
            cache_boundaries: vec![9, 10],
            selection_policy: Some("another".to_owned()),
            selection_accounting_ttl_ms: 11,
        };
        assert_eq!(
            policy_settings(&cache_aware),
            PolicySettings::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 2,
                balance_rel_threshold: 3.5,
                balance_prefix_margin_tokens: 4,
                balance_uncached_credit: true,
                eviction_interval_secs: 5,
                max_tree_size: 6,
                block_size: 7,
                balance_token_usage_threshold: 0.25,
                overload_token_usage_threshold: 0.75,
                overlap_decay: 0.125,
                selection_temperature: 0.0625,
                cache_index: IndexKind::Hash,
                cache_ttl_secs: 8,
                cache_boundaries: vec![9, 10],
                selection_policy: Some("another".to_owned()),
                selection_accounting_ttl_ms: 11,
            }
        );
        for (policy, expected) in [
            (PolicyConfig::Random, PolicySettings::Random),
            (PolicyConfig::RoundRobin, PolicySettings::RoundRobin),
            (PolicyConfig::Passthrough, PolicySettings::Passthrough),
            (
                PolicyConfig::ConsistentHashing,
                PolicySettings::ConsistentHashing,
            ),
            (
                PolicyConfig::LeastLoad {
                    kv_pressure_weight: 0.5,
                    mean_prefill_tokens: 2,
                    default_throughput: 3.5,
                },
                PolicySettings::LeastLoad {
                    kv_pressure_weight: 0.5,
                    mean_prefill_tokens: 2,
                    default_throughput: 3.5,
                },
            ),
            (
                PolicyConfig::Bucket {
                    balance_abs_threshold: 1,
                    balance_rel_threshold: 2.5,
                    bucket_adjust_interval_secs: 3,
                },
                PolicySettings::Bucket {
                    balance_abs_threshold: 1,
                    balance_rel_threshold: 2.5,
                    bucket_adjust_interval_secs: 3,
                },
            ),
            (
                PolicyConfig::PrefixHash {
                    prefix_token_count: 1,
                    load_factor: 2.5,
                    balance_abs_threshold: 3,
                    cache_boundaries: vec![4, 5],
                },
                PolicySettings::PrefixHash {
                    prefix_token_count: 1,
                    load_factor: 2.5,
                    balance_abs_threshold: 3,
                    cache_boundaries: vec![4, 5],
                },
            ),
        ] {
            assert_eq!(policy_settings(&policy), expected);
        }

        // Each value of the three enums the policies hold is its namesake.
        for (assignment_mode, assignment) in [
            (ManualAssignmentMode::Random, Assignment::Random),
            (ManualAssignmentMode::MinLoad, Assignment::MinLoad),
            (ManualAssignmentMode::MinGroup, Assignment::MinGroup),
            (ManualAssignmentMode::Delegate, Assignment::Delegate),
        ] {
            let manual = PolicyConfig::Manual {
                eviction_interval_secs: 1,
                max_idle_secs: 2,
                assignment_mode,
            };
            assert_eq!(
                policy_settings(&manual),
                PolicySettings::Manual {
                    eviction_interval_secs: 1,
                    max_idle_secs: 2,
                    assignment_mode: assignment,
                }
            );
        }
        for (load_metric, expected) in [
            (
                PowerOfTwoLoadMetric::ExpectedWait,
                LoadMetricSetting::ExpectedWait,
            ),
            (PowerOfTwoLoadMetric::Requests, LoadMetricSetting::Requests),
            (
                PowerOfTwoLoadMetric::LeastRequests,
                LoadMetricSetting::LeastRequests,
            ),
            (
                PowerOfTwoLoadMetric::ResidentTokens { request_tokens: 6 },
                LoadMetricSetting::ResidentTokens { request_tokens: 6 },
            ),
        ] {
            assert_eq!(
                policy_settings(&PolicyConfig::PowerOfTwo { load_metric }),
                PolicySettings::PowerOfTwo {
                    load_metric: expected
                }
            );
        }
        let tree = policy_settings(&at_flag_defaults("cache_aware"));
        assert!(matches!(
            tree,
            PolicySettings::CacheAware {
                cache_index: IndexKind::Tree,
                ..
            }
        ));
    }

    #[test]
    fn a_mode_reaches_the_policy_crate_with_its_side_policies_and_no_pool() {
        let mode = |mode: RoutingMode| config_of(mode).routing_settings().mode;
        assert_eq!(
            mode(RoutingMode::PrefillDecode {
                prefill_urls: vec![("http://prefill:8000".to_owned(), Some(9000))],
                decode_urls: vec!["http://decode:8000".to_owned()],
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy: None,
            }),
            ModeSettings::PrefillDecode {
                prefill_policy: Some(PolicySettings::RoundRobin),
                decode_policy: None,
            }
        );
        assert_eq!(
            mode(RoutingMode::EncodePrefillDecode {
                encode_urls: vec![("http://encode:8000".to_owned(), None)],
                prefill_urls: vec![("http://prefill:8000".to_owned(), None)],
                decode_urls: vec!["http://decode:8000".to_owned()],
                encode_policy: Some(PolicyConfig::Random),
                prefill_policy: None,
                decode_policy: Some(PolicyConfig::ConsistentHashing),
            }),
            ModeSettings::EncodePrefillDecode {
                encode_policy: Some(PolicySettings::Random),
                prefill_policy: None,
                decode_policy: Some(PolicySettings::ConsistentHashing),
            }
        );

        // A provider mode goes by its name in the configuration. Alone it
        // places by no policy and is refused by that name. Beside the
        // routers of self-hosted workers it reads as the regular mode.
        let regular = RouterConfig {
            enable_igw: true,
            ..RouterConfig::default()
        }
        .policy_document();
        assert!(regular.is_ok(), "{regular:?}");
        for (provider, name) in [
            (
                RoutingMode::OpenAI {
                    worker_urls: Vec::new(),
                },
                "openai",
            ),
            (
                RoutingMode::Anthropic {
                    worker_urls: Vec::new(),
                },
                "anthropic",
            ),
            (
                RoutingMode::Gemini {
                    worker_urls: Vec::new(),
                },
                "gemini",
            ),
        ] {
            let alone = config_of(provider);
            assert_eq!(alone.routing_settings().mode, ModeSettings::Provider(name));
            assert!(matches!(
                alone.policy_document(),
                Err(RoutingSettingsError::UnsupportedSetting { setting: "mode", value, .. }) if value == name
            ));
            let beside = RouterConfig {
                enable_igw: true,
                ..alone
            };
            assert_eq!(beside.policy_document(), regular, "{name}");
        }
    }

    fn config_of(mode: RoutingMode) -> RouterConfig {
        RouterConfig {
            mode,
            ..RouterConfig::default()
        }
    }

    #[test]
    fn the_default_selection_policy_is_left_out_and_another_is_refused_by_name() {
        let selecting = |named: &str| {
            let mut policy = at_flag_defaults("cache_aware");
            if let PolicyConfig::CacheAware {
                selection_policy, ..
            } = &mut policy
            {
                *selection_policy = Some(named.to_owned());
            }
            config(policy)
        };
        assert_eq!(
            selecting(DEFAULT_SELECTION_POLICY).policy_document(),
            Ok(preset("cache_aware"))
        );
        assert!(matches!(
            selecting("another").policy_document(),
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "selection_policy",
                ..
            })
        ));
    }
}
