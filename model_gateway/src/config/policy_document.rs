//! The policy document the gateway gives to its controller.
//!
//! The policy crate converts routing settings into the controller's
//! document: [`RoutingSettings`] says where a setting lands, what is refused
//! and what is left out. This module maps the gateway's configuration onto
//! that type, hands the gateway's own defaults to the comparison with a
//! supplied document, and validates a document against a table of stages
//! with the gateway's tenant header denied to it. No request path calls
//! this module.
//!
//! The routing settings are described twice: by this crate's configuration
//! types, which the flags, the configuration file and the bindings fill, and
//! by the policy crate's input type. The map below ties the two. It takes
//! each policy, each mode and the sticky override apart field by field, with
//! no `..`, so a setting added to one of them does not build until the map
//! carries it.

use smg_policy_controller::{
    document::{
        AppliedSpec, Assignment, ConnectionModeSetting, IndexKind, LoadMetricSetting, ModeSettings,
        OverloadSpec, Pairing, PolicySettings, PolicySpec, RoutingKeyOverrideSettings,
        RoutingSettings, RoutingSettingsError, ShadowSpec, SpecError,
    },
    stages::StageRegistry,
};
use smg_worker::identity::basic::ConnectionMode;

use super::{
    CacheIndexKind, GroupedDpSticky, ManualAssignmentMode, PdPairingMode, PolicyConfig,
    PowerOfTwoLoadMetric, RouterConfig, RoutingKeyOverrideConfig, RoutingMode,
};
use crate::policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY;

/// The overload veto's two words a gateway reads once for its process.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OverloadWords {
    /// Whether the gateway's default thresholds veto at all.
    pub(crate) protection: bool,
    /// Whether a request is shed where every worker it could use is vetoed.
    pub(crate) shed: bool,
}

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
    pub fn policy_document_in_force(&self) -> Result<PolicySpec, RoutingSettingsError> {
        let defaults = Self::default().routing_settings();
        self.routing_settings().policy_document_in_force(&defaults)
    }

    /// Whether the policy engine places this process's requests. It does
    /// where the policy controller is on, with one exception: a gateway
    /// whose every router is one provider's
    /// ([`RoutingSettings::provider_alone`]) places its requests by no
    /// routing policy. There the option moves nothing: no engine runs, the
    /// classic policies stay, and the start says so in one log line.
    ///
    /// Every reader of the option asks here: the start, the policy
    /// registry, the pairing mode in force and the validator's refusal of
    /// a document.
    ///
    /// A supplied document is never dropped in silence. Beside one the
    /// answer is yes under a provider's mode alone too, and the start then
    /// refuses the document by the mode's name
    /// ([`Self::policy_document_in_force`]). So a yes may still end in a
    /// refused start.
    pub(crate) fn policy_engine_decides(&self) -> bool {
        self.policy_controller
            && (self.policy_spec.is_some()
                || self.policy_spec_path.is_some()
                || self.routing_settings().provider_alone().is_none())
    }

    /// How strictly this gateway pairs a prefill with a decode. Where the
    /// policy controller runs a supplied document, it is that document's
    /// `plans.pd-pair.pairing`: the engine pairs within the classes the
    /// gateway derives and derives none itself, so the document's word is
    /// the mode they are derived under. Everywhere else it is the
    /// configuration's own mode, which the document the routing settings
    /// mean states too.
    pub(crate) fn pd_pairing_in_force(&self) -> PdPairingMode {
        match &self.policy_spec {
            Some(document) if self.policy_engine_decides() => {
                match document.plans.pd_pair.pairing {
                    Pairing::Off => PdPairingMode::Off,
                    Pairing::Lenient => PdPairingMode::Lenient,
                    Pairing::Strict => PdPairingMode::Strict,
                }
            }
            _ => self.pd_pairing_mode,
        }
    }

    /// The overload veto's two words in force: whether this gateway's
    /// default thresholds veto at all, and whether a request is shed where
    /// every worker it could use is vetoed. Where the policy controller
    /// runs a supplied document they are the words of that document's
    /// `overload` blocks ([`Self::supplied_overload_words`]): the worker
    /// objects are built with one default and the routes read one shed
    /// word, and the document's words are the ones they are built and read
    /// under. Everywhere else they are the configuration's own two
    /// settings, which the document the routing settings mean states too.
    /// They are those two as well for a document this gateway refuses,
    /// which no start runs.
    pub(crate) fn overload_words_in_force(&self) -> OverloadWords {
        let own = OverloadWords {
            protection: self.worker_overload_protection,
            shed: self.worker_overload_shed,
        };
        self.supplied_overload_words().ok().flatten().unwrap_or(own)
    }

    /// The two words of the `overload` blocks of the supplied document the
    /// policy controller runs, and `None` where it runs none. A worker
    /// object has one default set of thresholds and the routes read one
    /// shed word for the process, so the bindings say each word once. A
    /// binding with no block says the block's defaults, and so does a
    /// document with no binding.
    ///
    /// Two documents are refused, each in words a start ends on. One whose
    /// bindings differ on a word, by the path of the first binding that
    /// differs from the first, with both values. And one whose block is on
    /// while both of this gateway's thresholds are none: the switch would
    /// switch nothing on. The policy crate's comparison lets the two
    /// thresholds stand beside a document whatever they are, so the rule
    /// is this host's.
    pub(crate) fn supplied_overload_words(&self) -> Result<Option<OverloadWords>, String> {
        let document = match &self.policy_spec {
            Some(document) if self.policy_controller => document,
            _ => return Ok(None),
        };
        let stock = OverloadSpec::default();
        let mut blocks = document
            .bindings
            .iter()
            .map(|binding| binding.overload.as_ref().unwrap_or(&stock));
        let first = blocks.next().unwrap_or(&stock);
        for (at, block) in (1..).zip(blocks) {
            if block.enabled != first.enabled {
                return Err(overload_word_differs(
                    at,
                    "enabled",
                    block.enabled,
                    "a gateway builds its workers with one default set of overload thresholds",
                ));
            }
            if block.shed_when_all_vetoed != first.shed_when_all_vetoed {
                return Err(overload_word_differs(
                    at,
                    "shed_when_all_vetoed",
                    block.shed_when_all_vetoed,
                    "a gateway's routes read one shed word for the process",
                ));
            }
        }
        if first.enabled
            && self.worker_overload_waiting_requests.is_none()
            && self.worker_overload_token_usage.is_none()
        {
            return Err("worker_overload_waiting_requests and worker_overload_token_usage are both none beside a policy document whose overload block is on: a gateway with no default threshold vetoes by none; state a threshold, or switch the block off in the document".to_owned());
        }
        Ok(Some(OverloadWords {
            protection: first.enabled,
            shed: first.shed_when_all_vetoed,
        }))
    }

    /// Validates the document over `stages`, with the gateway's own
    /// headers denied to the view: the form an engine is built from.
    ///
    /// `stages` is the table of stages a document may name: the stock one
    /// ([`StageRegistry::builtin`]), or the table a build of the host's own
    /// hands in, which holds the stages that build added. A document that
    /// names a stage the table does not hold is refused by the stage's name.
    ///
    /// A binding's shadows by side are refused as not in this build: the
    /// engine this gateway hosts decides a shadow beside a single request
    /// alone, so a pair's would load and never run.
    pub fn hosted_policy(
        &self,
        document: &PolicySpec,
        stages: &StageRegistry,
    ) -> Result<AppliedSpec, SpecError> {
        let applied = AppliedSpec::validate_in(document, stages, &self.denied_view_headers())?;
        let by_side = document
            .bindings
            .iter()
            .position(|binding| matches!(binding.shadow, Some(ShadowSpec::Sides(_))));
        match by_side {
            Some(index) => Err(SpecError::NotInBuild {
                path: format!("bindings[{index}].shadow"),
                what: "a pair's shadow",
            }),
            None => Ok(applied),
        }
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
            connection_mode: match self.connection_mode {
                ConnectionMode::Http => ConnectionModeSetting::Http,
                ConnectionMode::Grpc => ConnectionModeSetting::Grpc,
                ConnectionMode::Zmq => ConnectionModeSetting::Zmq,
            },
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

/// The refusal of a supplied document whose binding `at` says `said` under
/// `field` of its `overload` block where the first binding says the other
/// word. `one` is why a gateway reads the word once.
fn overload_word_differs(at: usize, field: &str, said: bool, one: &str) -> String {
    format!(
        "bindings[{at}].overload.{field} = {said} is not the {} bindings[0] says, by its block or the block's default: {one}, so the bindings of its document say one word",
        !said
    )
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
    use smg_policy_controller::{
        document::PRESETS,
        engine::{Candidate, Constraints, Filter, RequestFacts},
    };

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

    /// A binding's `overload` block that says `enabled` and `shed`.
    fn block(enabled: bool, shed: bool) -> OverloadSpec {
        OverloadSpec {
            enabled,
            shed_when_all_vetoed: shed,
            ..OverloadSpec::default()
        }
    }

    /// The stock document with one binding for each of `blocks`, under that
    /// `overload` block or under none: the last binding is the stock one,
    /// and each binding ahead of it matches a model of its own.
    fn bound(blocks: &[Option<OverloadSpec>]) -> PolicySpec {
        let mut document = preset("round_robin");
        let stock = document.bindings.remove(0);
        let ahead = blocks.len().saturating_sub(1);
        document.bindings = blocks
            .iter()
            .enumerate()
            .map(|(at, block)| {
                let mut binding = stock.clone();
                if at < ahead {
                    binding.matcher.model = format!("model-{at}");
                }
                binding.overload = block.clone();
                binding
            })
            .collect();
        document
    }

    /// A gateway that runs `document` under the policy controller, with
    /// every routing setting at its default.
    fn supplied(document: PolicySpec) -> RouterConfig {
        RouterConfig {
            policy_controller: true,
            policy_spec: Some(document),
            ..RouterConfig::default()
        }
    }

    /// Where the policy controller runs a supplied document, the overload
    /// veto's two words are the ones its bindings say: the switch and the
    /// shed word, a binding with no block saying the block's defaults. With
    /// the option off, and with no document supplied, they are the
    /// configuration's own two settings.
    #[test]
    fn a_supplied_documents_overload_words_are_the_ones_in_force() {
        let words = |protection, shed| OverloadWords { protection, shed };
        let own = RouterConfig {
            worker_overload_protection: false,
            worker_overload_shed: true,
            ..RouterConfig::default()
        };
        assert_eq!(
            RouterConfig::default().overload_words_in_force(),
            words(true, false)
        );
        assert_eq!(own.overload_words_in_force(), words(false, true));
        assert_eq!(own.supplied_overload_words(), Ok(None));

        for (blocks, said) in [
            (vec![None], words(true, false)),
            (vec![Some(block(false, false))], words(false, false)),
            (vec![Some(block(true, true))], words(true, true)),
            (
                vec![Some(block(false, true)), Some(block(false, true))],
                words(false, true),
            ),
            // A binding with no block says the block's defaults.
            (
                vec![None, Some(block(true, false)), None],
                words(true, false),
            ),
        ] {
            let config = supplied(bound(&blocks));
            assert_eq!(
                config.supplied_overload_words(),
                Ok(Some(said)),
                "{blocks:?}"
            );
            assert_eq!(config.overload_words_in_force(), said, "{blocks:?}");
            // With the option off the document is not read.
            let off = RouterConfig {
                policy_controller: false,
                ..config
            };
            assert_eq!(off.overload_words_in_force(), words(true, false));
            assert_eq!(off.supplied_overload_words(), Ok(None));
        }
        // A document with no binding says the block's defaults.
        let mut bare = preset("round_robin");
        bare.bindings.clear();
        assert_eq!(
            supplied(bare).supplied_overload_words(),
            Ok(Some(words(true, false)))
        );
    }

    /// A worker object has one default set of thresholds and the routes
    /// read one shed word. Bindings that differ on either word are refused
    /// by the path of the first binding that differs from the first, with
    /// both values. For such a document, which no start runs, the words
    /// read are the configuration's own.
    #[test]
    fn bindings_that_differ_on_an_overload_word_are_refused_by_name() {
        let refused = |blocks: &[Option<OverloadSpec>]| {
            supplied(bound(blocks))
                .supplied_overload_words()
                .expect_err("the bindings differ")
        };
        assert_eq!(
            refused(&[None, Some(block(true, true)), Some(block(false, true))]),
            "bindings[1].overload.shed_when_all_vetoed = true is not the false bindings[0] says, by its block or the block's default: a gateway's routes read one shed word for the process, so the bindings of its document say one word"
        );
        assert_eq!(
            refused(&[Some(block(false, false)), Some(block(false, false)), None]),
            "bindings[2].overload.enabled = true is not the false bindings[0] says, by its block or the block's default: a gateway builds its workers with one default set of overload thresholds, so the bindings of its document say one word"
        );
        let differs = RouterConfig {
            worker_overload_shed: true,
            ..supplied(bound(&[Some(block(false, false)), None]))
        };
        assert_eq!(
            differs.overload_words_in_force(),
            OverloadWords {
                protection: true,
                shed: true
            }
        );
    }

    /// Both thresholds at none leave a gateway nothing to veto by. They stand
    /// beside a supplied document as every threshold does, so the comparison
    /// takes the document, whatever it is. This gateway then refuses one
    /// whose block is on, by the two settings' names and with the two ways
    /// out, and takes one whose bindings switch the block off. One threshold
    /// at none leaves the other to veto by.
    #[test]
    fn both_thresholds_at_none_are_refused_beside_a_document_whose_block_is_on() {
        let none = |document: PolicySpec| RouterConfig {
            worker_overload_waiting_requests: None,
            worker_overload_token_usage: None,
            ..supplied(document)
        };
        // A document of another policy than the default's: the comparison
        // names no difference, and the refusal names the two settings.
        let on = none(preset("round_robin"));
        assert_eq!(on.policy_document_in_force(), Ok(preset("round_robin")));
        assert_eq!(
            on.supplied_overload_words(),
            Err("worker_overload_waiting_requests and worker_overload_token_usage are both none beside a policy document whose overload block is on: a gateway with no default threshold vetoes by none; state a threshold, or switch the block off in the document".to_owned())
        );

        let off = bound(&[Some(block(false, false)), Some(block(false, false))]);
        let agreed = none(off.clone());
        assert_eq!(agreed.policy_document_in_force(), Ok(off));
        assert_eq!(
            agreed.supplied_overload_words(),
            Ok(Some(OverloadWords {
                protection: false,
                shed: false
            }))
        );

        let one = RouterConfig {
            worker_overload_token_usage: None,
            ..supplied(preset("round_robin"))
        };
        assert_eq!(one.policy_document_in_force(), Ok(preset("round_robin")));
        assert_eq!(
            one.supplied_overload_words(),
            Ok(Some(OverloadWords {
                protection: true,
                shed: false
            }))
        );
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
            connection_mode: ConnectionMode::Grpc,
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
        config.routing_key_override = RoutingKeyOverrideConfig {
            enabled: true,
            eviction_interval_secs: 17,
            max_idle_secs: 19,
            assignment_mode: ManualAssignmentMode::MinGroup,
            headers: vec!["x-session".to_owned()],
        };
        assert_eq!(
            config.routing_settings(),
            RoutingSettings {
                mode: ModeSettings::Regular,
                policy: PolicySettings::RoundRobin,
                enable_igw: true,
                connection_mode: ConnectionModeSetting::Grpc,
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
                    eviction_interval_secs: 17,
                    max_idle_secs: 19,
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

    /// The policy crate says the settings of a gateway whose configuration
    /// sets none ([`RoutingSettings::default`]). Its tests of the
    /// translation run over them, and over a literal of the cache-aware
    /// policy at its flags' defaults: `cache_aware()` in its
    /// `document/routing_settings.rs`. This gateway's default configuration
    /// means the first, and the literal below is the second. A default that
    /// moves on either side fails this case. The usage threshold is
    /// compared in 64 bits here: the crate's own cases compare it in 32.
    #[test]
    fn the_gateways_defaults_are_the_policy_crates_fixtures_of_them() {
        assert_eq!(
            RouterConfig::default().routing_settings(),
            RoutingSettings::default()
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

    /// The one test every reader of the option asks. The policy engine
    /// places the requests of every configuration that turns the policy
    /// controller on, a provider's mode alone apart: under the multi-router
    /// setting that mode has the regular document. With the option off the
    /// engine decides nowhere.
    #[test]
    fn the_policy_engine_decides_under_the_option_save_in_a_providers_mode_alone() {
        let modes = [
            RoutingMode::Regular {
                worker_urls: Vec::new(),
            },
            RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: None,
                decode_policy: None,
            },
            RoutingMode::EncodePrefillDecode {
                encode_urls: Vec::new(),
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            RoutingMode::OpenAI {
                worker_urls: Vec::new(),
            },
            RoutingMode::Anthropic {
                worker_urls: Vec::new(),
            },
            RoutingMode::Gemini {
                worker_urls: Vec::new(),
            },
        ];
        for mode in modes {
            let provider = matches!(
                mode,
                RoutingMode::OpenAI { .. }
                    | RoutingMode::Anthropic { .. }
                    | RoutingMode::Gemini { .. }
            );
            for enable_igw in [false, true] {
                let on = RouterConfig {
                    policy_controller: true,
                    enable_igw,
                    ..config_of(mode.clone())
                };
                let row = format!("{} with enable_igw = {enable_igw}", on.mode_type());
                assert_eq!(on.policy_engine_decides(), !provider || enable_igw, "{row}");
                let off = RouterConfig {
                    policy_controller: false,
                    ..on
                };
                assert!(!off.policy_engine_decides(), "{row}, the option off");
            }
        }
    }

    /// A supplied document is never dropped in silence. Beside one the
    /// answer is yes under a provider's mode alone too, and the document in
    /// force is refused there by the mode's name. The path of a document
    /// that was never loaded stands for a document.
    #[test]
    fn a_supplied_document_keeps_the_option_in_force_in_a_providers_mode_alone() {
        let alone = RouterConfig {
            policy_controller: true,
            ..config_of(RoutingMode::OpenAI {
                worker_urls: Vec::new(),
            })
        };
        assert!(!alone.policy_engine_decides());

        let supplied = RouterConfig {
            policy_spec: Some(preset("round_robin")),
            ..alone.clone()
        };
        assert!(supplied.policy_engine_decides());
        assert!(matches!(
            supplied.policy_document_in_force(),
            Err(RoutingSettingsError::UnsupportedSetting { setting: "mode", value, .. }) if value == "openai"
        ));

        let named = RouterConfig {
            policy_spec_path: Some("routing.yaml".to_owned()),
            ..alone
        };
        assert!(named.policy_engine_decides());
    }

    /// A pair gateway counts tokens on a prefill worker over HTTP alone: its
    /// document binds that single request there, behind the pair, and plans
    /// the pair alone over gRPC and over ZMQ, where no request would reach
    /// the binding.
    #[test]
    fn a_pair_gateways_document_binds_a_single_request_over_http_alone() {
        for (connection_mode, bindings) in [
            (ConnectionMode::Http, 2),
            (ConnectionMode::Grpc, 1),
            (ConnectionMode::Zmq, 1),
        ] {
            let config = RouterConfig {
                connection_mode,
                ..config_of(RoutingMode::PrefillDecode {
                    prefill_urls: Vec::new(),
                    decode_urls: Vec::new(),
                    prefill_policy: None,
                    decode_policy: None,
                })
            };
            let document = config.policy_document().expect("the mode's document");
            assert_eq!(document.bindings.len(), bindings, "{connection_mode:?}");
        }
    }

    /// The gateway keeps its tenant header to itself: a document that reads
    /// it is refused by the header's name, and the same document under
    /// another tenant header validates.
    #[test]
    fn a_document_that_reads_the_tenant_header_is_refused_by_its_name() {
        let mut keyed = config(at_flag_defaults("manual"));
        keyed.routing_key_override.headers = vec!["x-tenant-id".to_owned()];
        let document = keyed
            .policy_document()
            .expect("the manual policy's document");
        let stock = StageRegistry::builtin();
        assert!(keyed.hosted_policy(&document, &stock).is_ok());
        keyed.tenant_resolution.tenant_header_name = "x-tenant-id".to_owned();
        assert!(matches!(
            keyed.hosted_policy(&document, &stock).err(),
            Some(SpecError::DeniedHeader { name, .. }) if name == "x-tenant-id"
        ));
    }

    /// A filter of a host's own build: it keeps every worker.
    #[derive(Debug)]
    struct KeepsAll;

    impl Filter for KeepsAll {
        fn name(&self) -> &'static str {
            "keeps-all"
        }

        fn keep(&self, _: &RequestFacts<'_>, _: &Candidate, _: &Constraints<'_>) -> bool {
            true
        }
    }

    /// A document is validated against the table handed in: one that names
    /// a stage a host's own build added is taken over that build's table,
    /// and refused by the stage's name over the stock one.
    #[test]
    fn a_document_is_validated_against_the_table_of_stages_handed_in() {
        let mut stages = StageRegistry::builtin();
        let added = stages.filter("keeps-all", || Box::new(KeepsAll));
        assert!(added.is_ok(), "the name is free");
        let mut document = preset("random");
        let profile = document.profiles.get_mut("random");
        let profile = profile.expect("the preset's profile");
        profile.filter.push("keeps-all".into());

        let config = RouterConfig::default();
        assert!(config.hosted_policy(&document, &stages).is_ok());
        assert!(matches!(
            config.hosted_policy(&document, &StageRegistry::builtin()).err(),
            Some(SpecError::UnknownFilter { name, .. }) if name == "keeps-all"
        ));
    }

    /// One model's single binding with a shadow stands ahead of a pair's
    /// binding. The single binding's shadow loads, and a shadow by side on
    /// the pair's binding is refused by that binding's path.
    #[test]
    fn a_pair_bindings_shadow_is_refused_as_not_in_this_build() {
        let pairs = PolicySpec::pair_presets("round_robin", "round_robin");
        let mut document = pairs.expect("the presets of a pair");
        document.profiles.extend(preset("random").profiles);
        let mut single = preset("round_robin");
        document.profiles.append(&mut single.profiles);
        let mut ahead = single.bindings.pop().expect("the single binding");
        ahead.matcher.model = "one-model".to_owned();
        ahead.shadow = Some(ShadowSpec::Profile("random".to_owned()));
        document.bindings.insert(0, ahead);

        let config = RouterConfig::default();
        let refused = |document: &PolicySpec| {
            config
                .hosted_policy(document, &StageRegistry::builtin())
                .err()
        };
        assert_eq!(refused(&document), None);

        let pair = document.bindings.last_mut().expect("the pair's binding");
        let by_side = [("prefill".to_owned(), "random".to_owned())];
        pair.shadow = Some(ShadowSpec::Sides(by_side.into()));
        assert_eq!(
            refused(&document),
            Some(SpecError::NotInBuild {
                path: "bindings[1].shadow".to_owned(),
                what: "a pair's shadow",
            })
        );
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
