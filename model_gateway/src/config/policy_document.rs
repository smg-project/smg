//! The policy document the gateway gives to its controller.
//!
//! # Why this exists
//!
//! [`RouterConfig::policy_document`] converts the routing settings into the
//! controller's [`PolicySpec`]. At start-up, [`RouterConfig::policy_document_in_force`]
//! takes a supplied document when the settings are at their configuration
//! defaults or describe that same document. The store applies the document
//! in force at startup when the controller is selected. No request path
//! calls this module.
//!
//! # Where a setting lands
//!
//! - A policy name lands on the controller's preset of that name. A
//!   prefill-decode mode lands on the pair plan, with one profile a side.
//! - A gateway of many routers serves pairs and single workers over its
//!   one document, so the document binds both: pairs first, over the
//!   prefill and decode phases, and single workers behind. The sides of a
//!   pair run under the mode's side policies and single workers under the
//!   main policy, as on the classic line.
//! - A policy's own setting lands in the stage of the preset that reads it,
//!   and only where it differs from the number that stage runs at when its
//!   document names none. The controller names those numbers and this
//!   module reads them, so a configuration at them is the preset itself.
//! - The overload thresholds land on every binding, the pairing mode on
//!   the pair plan, the index settings under every model, and the block
//!   size and the cache boundaries in the view.
//! - The manual policy's settings land in the binding's session rule.
//! - A keyed policy's key sources land on the binding. The two rings name
//!   them under `routing_key`. The manual policy names them in its session
//!   rule while they are headers alone, and under `routing_key` once the
//!   override adds the request's own id.
//! - A number that means never lands as the largest its place holds: a
//!   balance threshold past the document's width, and the sweep interval of
//!   an index the configuration never sweeps. The classic trims an index at
//!   its sweep alone, so that index's size bound lands as the largest too.
//!
//! # What is refused
//!
//! A setting that decides a placement and has no place in the document is
//! refused by its name ([`PolicyDocumentError::Unsaid`]). It is never left
//! out in silence.
//!
//! # What is left out
//!
//! The document of many routers plans one family where the other has no
//! document. Pairs are left out of a regular configuration under the manual
//! policy, whose session rule runs under the single plan alone, and beside
//! the encode mode. Single workers are left out of a pair configuration
//! whose main policy reads a setting no document says. A request of the
//! family left out is refused, as it was.
//!
//! # What is not read
//!
//! What the gateway keeps for itself under the controller is no part of the
//! document: the pools and their addresses, discovery, the probes and the
//! load poll, retries and the breaker, the prefill queue and the decode
//! rooms. The pin of a sticky key to one engine of a grouped worker is the
//! dispatch's as well: it runs behind the decision, on the worker the
//! decision names. Its key is the gateway's own, read from the routing-key
//! headers. A supplied document names its own key sources, which need not
//! be that key, so the pin is refused beside one. The copy of the cache
//! boundaries at the top of the configuration decides nothing either: each
//! policy reads its own.

use serde::Serialize;
use serde_json::Value;
use smg_policy_controller::{
    affinity::{
        Gates, DEFAULT_THRESHOLD, DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY,
        DEFAULT_WARMUP_SECS, DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
    },
    bucket,
    builtins::{DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT},
    spec::{
        BestWithTemperatureParams, BindingSpec, BoundedLoadParams, BucketParams,
        ConsistentHashParams, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
        LeastLoadParams, OverloadSpec, Pairing, PickStage, PlanName, PrefixAffinityParams,
        ProfileSpec, RoutingKeySpec, ScoreSpec, ScoreStage, SpillParams, WarmupSliceParams,
        MANUAL_SESSION_HEADER, MAX_KEY_SOURCES, OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS,
    },
    AppliedSpec, Assignment, Phase, PolicySpec, PowerOfTwoMetric, SessionsSpec, SpecError,
};

use super::{
    CacheIndexKind, GroupedDpSticky, ManualAssignmentMode, PdPairingMode, PolicyConfig,
    PowerOfTwoLoadMetric, RouterConfig, RoutingMode,
};
use crate::policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY;

/// The encode side's policy where the configuration names none: encode
/// placement follows an item's content, not the request's main policy.
static ENCODE_DEFAULT: PolicyConfig = PolicyConfig::ConsistentHashing;

/// The encode side's profile, by its name in a document of three sides.
const ENCODE_SIDE: &str = "encode";

/// Why a routing configuration has no policy document to run.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum PolicyDocumentError {
    /// A setting decides a placement and the document has no place for it.
    #[error("{setting} = {value}: {reason}")]
    Unsaid {
        /// The setting, as the configuration spells its field.
        setting: &'static str,
        /// Its value, as it reads.
        value: String,
        /// Why no document says it.
        reason: &'static str,
    },
    /// A policy's preset holds no stage for one of the policy's settings:
    /// the controller's presets and this module have parted.
    #[error("the {policy} preset has no {stage} stage")]
    NoStage {
        policy: &'static str,
        stage: &'static str,
    },
    /// The same parting over a part that is no stage: a preset holds no
    /// profile, binding or session rule where a setting is written.
    #[error("the {policy} preset has no {part}")]
    NoPart {
        policy: &'static str,
        part: &'static str,
    },
    /// A failure to produce or validate a policy document.
    #[error(transparent)]
    Preset(#[from] SpecError),
    /// The flags change the default document and differ from the supplied one.
    #[error("the routing flags and the policy document disagree at {at}: the flags mean {flags}, the document says {document}")]
    Disagree {
        /// Where the documents first differ, in their own field names.
        at: String,
        /// The value from the routing flags.
        flags: String,
        /// The value from the supplied document.
        document: String,
    },
}

fn unsaid(
    setting: &'static str,
    value: impl ToString,
    reason: &'static str,
) -> PolicyDocumentError {
    PolicyDocumentError::Unsaid {
        setting,
        value: value.to_string(),
        reason,
    }
}

impl RouterConfig {
    /// The policy document this configuration says, or the first of its
    /// settings that no document says.
    ///
    /// A pure function of the configuration. The document is not validated
    /// here: the controller validates a document where it is applied, and
    /// names there a stage its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, PolicyDocumentError> {
        refuse_unsaid(self)?;
        if let Some(both) = both_families(self)? {
            match written(self, both) {
                // A setting the other family's policy reads and no document
                // says leaves that family out. The document then plans the
                // configuration's own family alone.
                Err(PolicyDocumentError::Unsaid { .. }) => {}
                said => return said,
            }
        }
        written(self, presets(self)?)
    }

    /// [`Self::policy_document`] with one binding more for each model of
    /// `labelled`: a model and the name of the policy its first worker's
    /// `policy` label gave it. The classic line builds such a model's policy
    /// from its name alone, so the binding runs that policy's preset, at the
    /// numbers the preset itself runs at. It matches the model's id alone,
    /// and stands ahead of the binding of every other single worker's
    /// request and behind a pair's, in the order the models are handed.
    ///
    /// The whole plan is written as the settings' own is, so a label's
    /// policy shares the places the document holds once with every other
    /// side. A label that names the main policy binds nothing: the settings
    /// write that policy already. A label the document cannot say is left
    /// out, and its model is decided as every other is. Each is answered
    /// with why: a policy that names another value where the document holds
    /// one for every side, a name no preset has, a model's id a binding
    /// reads as a pattern, and a document that plans no single workers. A
    /// label moves no family in or out: the families are settled with none.
    ///
    /// Each document with one label more is checked as the door checks a
    /// document this gateway hosts ([`Self::hosted_policy`]). A label the
    /// door would refuse is left out alone, with the door's reason, so it
    /// never fails the documents of the labels behind it.
    pub(crate) fn policy_document_labelled(
        &self,
        labelled: &[(String, &'static str)],
    ) -> Result<(PolicySpec, Vec<(String, PolicyDocumentError)>), PolicyDocumentError> {
        let settled = self.policy_document()?;
        let both = both_families(self)?
            .is_some_and(|(both, _)| both.bindings.len() == settled.bindings.len());
        let mut document = settled.clone();
        let mut bound: Vec<(&str, PolicyConfig)> = Vec::with_capacity(labelled.len());
        let mut left = Vec::new();
        for (model, name) in labelled {
            if settled.profiles.contains_key(*name) {
                continue;
            }
            let policy = match labelled_policy(model, name, &settled) {
                Ok(policy) => policy,
                Err(why) => {
                    left.push((model.clone(), why));
                    continue;
                }
            };
            bound.push((model, policy));
            let planned = match both_families(self)? {
                Some(planned) if both => planned,
                _ => presets(self)?,
            };
            let candidate = bind_labelled(planned, &bound)
                .and_then(|planned| written(self, planned))
                .and_then(|written| {
                    self.hosted_policy(&written)?;
                    Ok(written)
                });
            match candidate {
                Ok(written) => document = written,
                Err(why) => {
                    bound.pop();
                    left.push((model.clone(), why));
                }
            }
        }
        Ok((document, left))
    }

    /// The document this gateway runs. Without a supplied document, it is
    /// the one the routing settings mean. A supplied document takes effect
    /// if the settings describe the same document or the default
    /// configuration's document for this mode. A conflict names the first
    /// difference and the two values.
    ///
    /// Both launchers refuse a routing setting beside a document before
    /// this runs, so the comparison judges a configuration built by hand.
    /// The generation is no part of it: it numbers the document, and no
    /// routing setting says it. A document that names none is this
    /// gateway's first and takes number 1, since a store applies only a
    /// number above the one it holds.
    pub fn policy_document_in_force(&self) -> Result<PolicySpec, PolicyDocumentError> {
        let mut said = self.policy_document()?;
        let Some(given) = &self.policy_spec else {
            return Ok(said);
        };
        // The dispatch pins a sticky key by the gateway's own key. A supplied
        // document names its own key sources: where they are not that key,
        // the pin would split a conversation the document keeps together.
        if self.grouped_dp_sticky != GroupedDpSticky::Off {
            return Err(unsaid(
                "grouped_dp_sticky",
                self.grouped_dp_sticky.as_str(),
                "beside a supplied document the pin would read the gateway's own key, which that document's key sources need not be",
            ));
        }
        let mut given = given.clone();
        if given.generation == 0 {
            given.generation = 1;
        }
        // The default configuration of this gateway's kind: a provider mode
        // has a document under the multi-router setting alone.
        let plain = RouterConfig {
            mode: self.mode.with_defaults(),
            enable_igw: self.enable_igw,
            ..RouterConfig::default()
        }
        .policy_document()?;
        if said == plain {
            return Ok(given);
        }
        said.generation = given.generation;
        if said == given {
            return Ok(given);
        }

        let written = |spec| {
            serde_json::to_value(spec)
                .map_err(|error| SpecError::NoCanonicalForm(error.to_string()))
        };
        let flags = written(&said)?;
        let document = written(&given)?;
        if let Some(difference) = first_difference(Some(&flags), Some(&document), String::new()) {
            return Err(difference);
        }

        // A field the reader accepts to report a validation error can be
        // absent from the written form. Let the document name that error.
        given.validate()?;
        Err(PolicyDocumentError::Disagree {
            at: "document".to_owned(),
            flags: flags.to_string(),
            document: document.to_string(),
        })
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
}

impl RoutingMode {
    /// Start with empty pools and clear any side policies, so the default
    /// document for this mode carries none of the configured side settings.
    fn with_defaults(&self) -> Self {
        match self {
            Self::Regular { .. } => Self::Regular {
                worker_urls: vec![],
            },
            Self::PrefillDecode { .. } => Self::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: None,
                decode_policy: None,
            },
            Self::EncodePrefillDecode { .. } => Self::EncodePrefillDecode {
                encode_urls: vec![],
                prefill_urls: vec![],
                decode_urls: vec![],
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            Self::OpenAI { .. } => Self::OpenAI {
                worker_urls: vec![],
            },
            Self::Anthropic { .. } => Self::Anthropic {
                worker_urls: vec![],
            },
            Self::Gemini { .. } => Self::Gemini {
                worker_urls: vec![],
            },
        }
    }
}

/// Walk each object in the flags' written order, then through keys that
/// appear only in the supplied document. Walk lists by index.
fn first_difference(
    flags: Option<&Value>,
    document: Option<&Value>,
    at: String,
) -> Option<PolicyDocumentError> {
    match (flags, document) {
        (Some(Value::Object(flags)), Some(Value::Object(document))) => flags
            .keys()
            .chain(document.keys().filter(|key| !flags.contains_key(*key)))
            .find_map(|key| {
                let path = if at.is_empty() {
                    key.clone()
                } else {
                    format!("{at}.{key}")
                };
                first_difference(flags.get(key), document.get(key), path)
            }),
        (Some(Value::Array(flags)), Some(Value::Array(document))) => {
            (0..flags.len().max(document.len())).find_map(|index| {
                first_difference(
                    flags.get(index),
                    document.get(index),
                    format!("{at}[{index}]"),
                )
            })
        }
        _ if flags == document => None,
        _ => {
            let written = |value: Option<&Value>| {
                value.map_or_else(|| "nothing".to_owned(), Value::to_string)
            };
            Some(PolicyDocumentError::Disagree {
                at,
                flags: written(flags),
                document: written(document),
            })
        }
    }
}

/// The settings that decide a placement today and that no document says.
/// The policy of a preset's name with every setting at its flag's default:
/// the numbers that preset runs at. `None` where no preset has the name.
fn at_flag_defaults(name: &str) -> Option<PolicyConfig> {
    Some(match name {
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
        _ => return None,
    })
}

/// A worker's `policy` label, as a refusal names it.
const POLICY_LABEL: &str = "worker label policy";

/// The policy a label's name means, at the numbers its preset runs at, or
/// why `settled`, the settings' document, binds no such model of its own.
fn labelled_policy(
    model: &str,
    name: &'static str,
    settled: &PolicySpec,
) -> Result<PolicyConfig, PolicyDocumentError> {
    let single = settled
        .bindings
        .last()
        .is_some_and(|last| !matches!(last.plan, Some(PlanName::PdPair | PlanName::Epd)));
    if !single {
        return Err(unsaid(
            POLICY_LABEL,
            name,
            "the document plans no single workers, and a side's policy is the mode's",
        ));
    }
    if model.is_empty() || model.contains('*') {
        return Err(unsaid(
            POLICY_LABEL,
            name,
            "a binding matches no model of this id alone: it reads the id as a pattern",
        ));
    }
    at_flag_defaults(name).ok_or_else(|| unsaid(POLICY_LABEL, name, "no preset has this name"))
}

/// A plan with one binding more for each labelled model: the model's id
/// alone under the preset of its policy's name, ahead of the plan's last
/// binding, the one of every other single worker's request. A preset's
/// profile is taken once, however many models name it.
fn bind_labelled<'a>(
    (mut document, mut sides): Planned<'a>,
    labelled: &'a [(&str, PolicyConfig)],
) -> Result<Planned<'a>, PolicyDocumentError> {
    for (model, policy) in labelled {
        let name = policy.name();
        let mut preset = PolicySpec::preset(name)?;
        let mut binding = preset.bindings.pop().ok_or(PolicyDocumentError::NoPart {
            policy: name,
            part: "binding",
        })?;
        (*model).clone_into(&mut binding.matcher.model);
        for (profile, written) in preset.profiles {
            document.profiles.entry(profile).or_insert(written);
        }
        let last = document.bindings.len().saturating_sub(1);
        document.bindings.insert(last, binding);
        sides.insert(last, vec![(name, policy)]);
    }
    Ok((document, sides))
}

fn refuse_unsaid(config: &RouterConfig) -> Result<(), PolicyDocumentError> {
    if config.dp_minimum_tokens_scheduler {
        return Err(unsaid(
            "dp_minimum_tokens_scheduler",
            true,
            "the document has no rank policy: under the controller a rank is a candidate of the profile",
        ));
    }
    Ok(())
}

/// A profile of the document and the policy that decides it.
type Side<'a> = (&'static str, &'a PolicyConfig);

/// A document and, binding by binding, the sides its plan runs.
type Planned<'a> = (PolicySpec, Vec<Vec<Side<'a>>>);

/// The presets the mode's policies name, as a document of one binding, with
/// its profiles: one named for the policy under the regular mode, one a side
/// under the pair and the encode modes. A provider mode of a multi-router
/// gateway reads as the regular mode: the document says the policy of its
/// self-hosted workers. The pools of a mode are the gateway's and are not
/// read.
fn presets(config: &RouterConfig) -> Result<Planned<'_>, PolicyDocumentError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (prefill, decode) = (mode.get_prefill_policy(main), mode.get_decode_policy(main));
    let regular = || -> Result<_, PolicyDocumentError> {
        Ok((PolicySpec::preset(main.name())?, vec![(main.name(), main)]))
    };
    let (document, sides) = match mode {
        RoutingMode::Regular { worker_urls: _ } => regular()?,
        RoutingMode::PrefillDecode {
            prefill_urls: _,
            decode_urls: _,
            prefill_policy: _,
            decode_policy: _,
        } => (
            PolicySpec::pair_presets(prefill.name(), decode.name())?,
            vec![("prefill", prefill), ("decode", decode)],
        ),
        RoutingMode::EncodePrefillDecode {
            encode_urls: _,
            prefill_urls: _,
            decode_urls: _,
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => {
            let encode = mode.get_encode_policy(&ENCODE_DEFAULT);
            (
                PolicySpec::epd_presets(encode.name(), prefill.name(), decode.name())?,
                vec![
                    (ENCODE_SIDE, encode),
                    ("prefill", prefill),
                    ("decode", decode),
                ],
            )
        }
        RoutingMode::OpenAI { worker_urls: _ }
        | RoutingMode::Anthropic { worker_urls: _ }
        | RoutingMode::Gemini { worker_urls: _ } => match config.provider_alone() {
            Some(provider) => return Err(provider_mode(provider)),
            None => regular()?,
        },
    };
    Ok((document, vec![sides]))
}

/// The document of a gateway of many routers, which serves pairs and single
/// workers side by side: the pair binding first, over the prefill and decode
/// phases, and the single binding behind it for every other request. Each
/// family runs under the policies the classic line gives it. The sides of a
/// pair run under the mode's side policies, which are the main policy in a
/// regular configuration. Single workers run under the main policy.
///
/// `None` where the document plans one family: a gateway of one router, the
/// encode mode, and a regular configuration whose main policy runs on no
/// side of a pair. The manual policy is that one: its picker leaves the
/// placement to the binding's session rule, and a rule runs under the single
/// plan alone.
fn both_families(config: &RouterConfig) -> Result<Option<Planned<'_>>, PolicyDocumentError> {
    let (mode, main) = (&config.mode, &config.policy);
    let pairs = match mode {
        RoutingMode::Regular { worker_urls: _ }
        | RoutingMode::OpenAI { worker_urls: _ }
        | RoutingMode::Anthropic { worker_urls: _ }
        | RoutingMode::Gemini { worker_urls: _ } => !matches!(main, PolicyConfig::Manual { .. }),
        RoutingMode::PrefillDecode {
            prefill_urls: _,
            decode_urls: _,
            prefill_policy: _,
            decode_policy: _,
        } => true,
        RoutingMode::EncodePrefillDecode {
            encode_urls: _,
            prefill_urls: _,
            decode_urls: _,
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => false,
    };
    if !(config.enable_igw && pairs) {
        return Ok(None);
    }
    let (prefill, decode) = (mode.get_prefill_policy(main), mode.get_decode_policy(main));
    let mut document = PolicySpec::pair_presets(prefill.name(), decode.name())?;
    for binding in &mut document.bindings {
        binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
    }
    let single = PolicySpec::preset(main.name())?;
    document.profiles.extend(single.profiles);
    document.bindings.extend(single.bindings);
    let sides = vec![
        vec![("prefill", prefill), ("decode", decode)],
        vec![(main.name(), main)],
    ];
    Ok(Some((document, sides)))
}

/// Writes the configuration's settings over a planned document: each side's
/// policy into its profile, what the sides of a binding share into that
/// binding, and what every side shares into the places the document holds
/// once. The overload block lands on every binding, so one set of thresholds
/// judges the pools of each family.
fn written<'a>(
    config: &'a RouterConfig,
    (mut document, bindings): Planned<'a>,
) -> Result<PolicySpec, PolicyDocumentError> {
    // A plan of several sides runs the pair: the pairing mode is its rule.
    if bindings.iter().any(|sides| sides.len() > 1) {
        document.plans.pd_pair.pairing = match config.pd_pairing_mode {
            PdPairingMode::Off => Pairing::Off,
            PdPairingMode::Lenient => Pairing::Lenient,
            PdPairingMode::Strict => Pairing::Strict,
        };
    }
    let mut shared = Shared::default();
    let mut bound = Vec::with_capacity(bindings.len());
    for sides in &bindings {
        for &(side, policy) in sides {
            let profile =
                document
                    .profiles
                    .get_mut(side)
                    .ok_or_else(|| PolicyDocumentError::NoPart {
                        policy: policy.name(),
                        part: "profile",
                    })?;
            write_policy(policy, config, profile, &mut shared)?;
            // The encode side places an item by the item's content: it reads
            // no key, and the override never places it.
            if side != ENCODE_SIDE {
                if let Some(sources) = key_sources(policy, config)? {
                    shared.bound.keys.push((policy.name(), sources));
                }
            }
        }
        bound.push((sides.len(), std::mem::take(&mut shared.bound)));
    }
    shared.write(&mut document)?;
    if document.bindings.is_empty() || document.bindings.len() != bound.len() {
        return Err(PolicyDocumentError::NoPart {
            policy: config.policy.name(),
            part: "binding",
        });
    }
    for (binding, (sides, bound)) in document.bindings.iter_mut().zip(bound) {
        bound.write(config, sides, binding, &mut document.view.headers)?;
    }
    let overload = overload(config)?;
    for binding in &mut document.bindings {
        binding.overload.clone_from(&overload);
    }
    Ok(document)
}

fn provider_mode(mode: &'static str) -> PolicyDocumentError {
    unsaid(
        "mode",
        mode,
        "a provider mode places its requests by no routing policy",
    )
}

/// Writes one side's policy over its preset: its settings into the stages of
/// its profile, and into the places the document's sides share.
fn write_policy<'a>(
    policy: &'a PolicyConfig,
    config: &RouterConfig,
    profile: &mut ProfileSpec,
    shared: &mut Shared<'a>,
) -> Result<(), PolicyDocumentError> {
    let name = policy.name();
    match policy {
        PolicyConfig::Random
        | PolicyConfig::RoundRobin
        | PolicyConfig::Passthrough
        | PolicyConfig::ConsistentHashing => Ok(()),
        PolicyConfig::PowerOfTwo { load_metric } => {
            let metric = match *load_metric {
                PowerOfTwoLoadMetric::ExpectedWait => PowerOfTwoMetric::ExpectedWait,
                PowerOfTwoLoadMetric::Requests => PowerOfTwoMetric::Requests,
                PowerOfTwoLoadMetric::LeastRequests => PowerOfTwoMetric::LeastRequests,
                PowerOfTwoLoadMetric::ResidentTokens { request_tokens } => {
                    PowerOfTwoMetric::ResidentTokens {
                        request_tokens: narrow("load_metric.request_tokens", request_tokens)?,
                    }
                }
            };
            // The preset in its metric's form: the form's scorers and picker
            // under the side's own filters.
            let form = PolicySpec::power_of_two(metric)?
                .profiles
                .remove(name)
                .ok_or(PolicyDocumentError::NoPart {
                    policy: name,
                    part: "profile",
                })?;
            (profile.score, profile.pick) = (form.score, form.pick);
            Ok(())
        }
        PolicyConfig::LeastLoad {
            kv_pressure_weight,
            mean_prefill_tokens,
            default_throughput,
        } => {
            let written = LeastLoadParams {
                kv_pressure_weight: unless(*kv_pressure_weight as f32, DEFAULT_KV_PRESSURE_WEIGHT),
                mean_prefill_tokens: unless(*mean_prefill_tokens, DEFAULT_MEAN_PREFILL_TOKENS),
                default_throughput: unless(*default_throughput as f32, DEFAULT_THROUGHPUT),
                ..LeastLoadParams::default()
            };
            if written != LeastLoadParams::default() {
                restage(profile, name, "least-load", written.weight, |weight| {
                    ScoreStage::LeastLoad(LeastLoadParams { weight, ..written })
                })?;
            }
            Ok(())
        }
        PolicyConfig::Bucket {
            balance_abs_threshold,
            balance_rel_threshold,
            bucket_adjust_interval_secs,
        } => {
            let written = BucketParams {
                abs_threshold: unless(
                    threshold(*balance_abs_threshold),
                    bucket::DEFAULT_ABS_THRESHOLD,
                ),
                rel_threshold: unless(*balance_rel_threshold, bucket::DEFAULT_REL_THRESHOLD),
                adjust_interval_s: unless(
                    narrow("bucket_adjust_interval_secs", *bucket_adjust_interval_secs)?,
                    bucket::DEFAULT_ADJUST_INTERVAL_S,
                ),
                ..BucketParams::default()
            };
            if written != BucketParams::default() {
                restage(profile, name, "bucket", written.weight, |weight| {
                    ScoreStage::Bucket(BucketParams { weight, ..written })
                })?;
            }
            Ok(())
        }
        PolicyConfig::Manual {
            eviction_interval_secs,
            max_idle_secs,
            assignment_mode,
        } => {
            // The manual preset places by the binding's session rule, which
            // every side shares: its settings land there.
            shared.bound.manual_sides += 1;
            share(
                &mut shared.bound.manual,
                (*eviction_interval_secs, *max_idle_secs, *assignment_mode),
                "eviction_interval_secs, max_idle_secs, assignment_mode",
            )
        }
        PolicyConfig::PrefixHash {
            prefix_token_count,
            load_factor,
            balance_abs_threshold,
            cache_boundaries,
        } => {
            // The preset names the ring's three numbers itself, so they are
            // written whatever they are.
            let prefix_tokens = Some(narrow("prefix_token_count", *prefix_token_count)?);
            let bounded_load = Some(BoundedLoadParams {
                factor: Some(*load_factor),
                abs_margin: Some(threshold(*balance_abs_threshold)),
            });
            let bare = ConsistentHashParams::default().weight;
            restage(profile, name, "consistent-hash", bare, |weight| {
                ScoreStage::ConsistentHash(ConsistentHashParams {
                    weight,
                    prefix_tokens,
                    bounded_load,
                })
            })?;
            share(
                &mut shared.boundaries,
                cache_boundaries.as_slice(),
                "cache_boundaries",
            )
        }
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
        } => {
            if let Some(named) = selection_policy
                .as_deref()
                .filter(|named| *named != DEFAULT_SELECTION_POLICY)
            {
                return Err(unsaid(
                    "selection_policy",
                    named,
                    "the document names a picker, and no selection policy but the default",
                ));
            }
            // The gates a silent profile runs under, as the controller builds
            // them.
            let gates = Gates::default();
            let spill = SpillParams {
                abs: unless(threshold(*balance_abs_threshold), gates.abs),
                rel: unless(*balance_rel_threshold, gates.rel),
                prefix_margin_tokens: unless(
                    *balance_prefix_margin_tokens,
                    gates.prefix_margin_tokens,
                ),
                uncached_credit: unless(*balance_uncached_credit, false),
            };
            let affinity = PrefixAffinityParams {
                threshold: unless(*cache_threshold, DEFAULT_THRESHOLD),
                decay: unless(*overlap_decay, 0.0),
                accounting_ttl_ms: unless(*selection_accounting_ttl_ms, 0),
                spill: unless(spill, SpillParams::default()),
                ..PrefixAffinityParams::default()
            };
            if affinity != PrefixAffinityParams::default() {
                restage(
                    profile,
                    name,
                    "prefix-affinity",
                    affinity.weight,
                    |weight| {
                        ScoreStage::PrefixAffinity(PrefixAffinityParams { weight, ..affinity })
                    },
                )?;
            }
            let pressure = KvPressureParams {
                spread: unless(*balance_token_usage_threshold, gates.kv_spread),
                ceiling: unless(*overload_token_usage_threshold, gates.kv_ceiling),
                ..KvPressureParams::default()
            };
            if pressure != KvPressureParams::default() {
                restage(profile, name, "kv-pressure", pressure.weight, |weight| {
                    ScoreStage::KvPressure(KvPressureParams { weight, ..pressure })
                })?;
            }
            let warmup = WarmupSliceParams {
                secs: unless(config.worker_warmup_secs, DEFAULT_WARMUP_SECS),
                share: unless(config.worker_warmup_share, DEFAULT_WARMUP_SHARE),
                blocks: unless(
                    narrow("worker_warmup_blocks", config.worker_warmup_blocks)?,
                    DEFAULT_WARMUP_BLOCKS,
                ),
                thin_ratio: unless(config.worker_warmup_thin_ratio, DEFAULT_WARMUP_THIN_RATIO),
                divert_every: unless(
                    config.worker_warmup_divert_every,
                    DEFAULT_WARMUP_DIVERT_EVERY,
                ),
                // The slice counts a hit as a miss by the affinity's own
                // threshold: the one setting is written in both stages.
                miss_share: unless(*cache_threshold, DEFAULT_THRESHOLD),
                ..WarmupSliceParams::default()
            };
            if warmup != WarmupSliceParams::default() {
                restage(profile, name, "warmup-slice", warmup.weight, |weight| {
                    ScoreStage::WarmupSlice(WarmupSliceParams { weight, ..warmup })
                })?;
            }
            if *selection_temperature != 0.0 {
                profile.pick = PickStage::BestWithTemperature(BestWithTemperatureParams {
                    temperature: *selection_temperature,
                })
                .into();
            }
            share(&mut shared.block_size, *block_size, "block_size")?;
            share(
                &mut shared.boundaries,
                cache_boundaries.as_slice(),
                "cache_boundaries",
            )?;
            share(
                &mut shared.index,
                (
                    *cache_index,
                    narrow("cache_ttl_secs", *cache_ttl_secs)?,
                    narrow("max_tree_size", *max_tree_size)?,
                    narrow("eviction_interval_secs", *eviction_interval_secs)?,
                ),
                "cache_index, cache_ttl_secs, max_tree_size, eviction_interval_secs",
            )
        }
    }
}

/// Puts a stage in the place of the profile's stage of that name, at the
/// weight the preset gives it. A stage the preset names bare weighs what its
/// parameters default to in the controller: the caller hands that as `bare`.
fn restage(
    profile: &mut ProfileSpec,
    policy: &'static str,
    stage: &'static str,
    bare: f32,
    written: impl FnOnce(f32) -> ScoreStage,
) -> Result<(), PolicyDocumentError> {
    let missing = PolicyDocumentError::NoStage { policy, stage };
    let Some(entry) = profile.score.iter_mut().find(|entry| entry.name() == stage) else {
        return Err(missing);
    };
    let weight = match entry {
        ScoreSpec::Name(_) => bare,
        ScoreSpec::Params(preset) => preset.weight(),
        ScoreSpec::Added { .. } => return Err(missing),
    };
    *entry = written(weight).into();
    Ok(())
}

/// `value` where it differs from `silent`, what the document means where
/// it says nothing.
fn unless<T: PartialEq>(value: T, silent: T) -> Option<T> {
    (value != silent).then_some(value)
}

/// A number of the configuration in the document's width.
fn narrow<T, U>(setting: &'static str, value: U) -> Result<T, PolicyDocumentError>
where
    T: TryFrom<U>,
    U: Copy + ToString,
{
    T::try_from(value).map_err(|_| unsaid(setting, value, "past the document's range"))
}

/// A balance threshold in the document's width. One past the width lands as
/// the width's largest number: no load reaches either, so the stage that
/// reads it decides the same under both.
fn threshold(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Seconds of a session rule in the document's width. The rule reads its
/// largest value as for ever, so a setting that large has no place in it.
fn seconds(setting: &'static str, secs: u64) -> Result<u32, PolicyDocumentError> {
    narrow(setting, secs).and_then(|narrowed: u32| {
        unless(narrowed, u32::MAX).ok_or_else(|| {
            unsaid(
                setting,
                secs,
                "the document reads this many seconds as for ever",
            )
        })
    })
}

/// What the sides of a document share. The document holds each once, so the
/// first side to name one sets it, and a side that names another value is
/// refused.
#[derive(Default)]
struct Shared<'a> {
    block_size: Option<usize>,
    boundaries: Option<&'a [usize]>,
    /// The index's kind, entry lifetime, size and sweep interval.
    index: Option<(CacheIndexKind, u32, u64, u32)>,
    /// What the sides of the binding being written share.
    bound: Bound,
}

/// What the sides of one binding share. The binding holds each once.
#[derive(Default)]
struct Bound {
    /// The manual policy's sweep interval, idle time and assignment.
    manual: Option<(u64, u64, ManualAssignmentMode)>,
    manual_sides: usize,
    /// The sources each keyed side reads a request's key from, under the
    /// side's policy.
    keys: Vec<(&'static str, RoutingKeySpec)>,
}

/// A value as the configuration spells it.
fn spelled<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|unspelled| unspelled.to_string())
}

fn share<T: PartialEq + Serialize>(
    held: &mut Option<T>,
    value: T,
    setting: &'static str,
) -> Result<(), PolicyDocumentError> {
    match held {
        Some(held) if *held != value => Err(unsaid(
            setting,
            format!("{} and {}", spelled(held), spelled(&value)),
            "two sides name two values and the document holds one for every side",
        )),
        Some(_) | None => {
            *held = Some(value);
            Ok(())
        }
    }
}

impl Shared<'_> {
    /// Writes what the sides named into the places the document holds once.
    fn write(self, document: &mut PolicySpec) -> Result<(), PolicyDocumentError> {
        if let Some(block_size) = self.block_size {
            document.view.block_size = unless(
                narrow("block_size", block_size)?,
                Gates::default().block_size,
            );
        }
        if let Some(boundaries) = self.boundaries {
            document.view.boundaries = boundaries
                .iter()
                .map(|at| narrow("cache_boundaries", *at))
                .collect::<Result<_, _>>()?;
        }
        let index = self.index.map(|(kind, ttl_s, max_size, sweep_s)| {
            // The classic trims its index at its sweep alone, and sweeps
            // nothing at an interval of zero: that index is never trimmed.
            // The document's cadence and size both start at one, so its
            // longest cadence and its largest size say it. No sweep falls
            // inside a process's life, and no write meets the bound.
            let (max_size, sweep_s) = if sweep_s == 0 {
                (u64::MAX, u32::MAX)
            } else {
                (max_size, sweep_s)
            };
            IndexSpec {
                kind: match kind {
                    CacheIndexKind::Tree => IndexKind::Tree,
                    CacheIndexKind::Hash => IndexKind::Hash,
                },
                ttl_s,
                max_size,
                sweep_s,
            }
        });
        if let Some(index) = index.and_then(|index| unless(index, IndexSpec::default())) {
            document.index.insert("*".to_string(), index);
        }
        Ok(())
    }
}

impl Bound {
    /// Writes what the sides named into their binding, a plan of `sides`
    /// profiles. The headers the manual policy reads are the view's too.
    fn write(
        self,
        config: &RouterConfig,
        sides: usize,
        binding: &mut BindingSpec,
        view_headers: &mut Vec<String>,
    ) -> Result<(), PolicyDocumentError> {
        let Some((eviction_interval_secs, max_idle_secs, assignment_mode)) = self.manual else {
            // A binding names its key's sources once: what its keyed sides
            // read, or nothing, over whatever its presets wrote.
            let key = one_list(self.keys)?.filter(|key| !key.sources.is_empty());
            if let Some(key) = &key {
                within_a_block(key, config)?;
            }
            binding.routing_key = key;
            return Ok(());
        };
        if self.manual_sides != sides {
            return Err(unsaid(
                "policy",
                "manual",
                "the manual policy places by the binding's one session rule, so every side of a plan is manual or none is",
            ));
        }
        let key = one_list(self.keys)?.ok_or(PolicyDocumentError::NoPart {
            policy: "manual",
            part: "key sources",
        })?;
        let headers: Vec<String> = key
            .sources
            .iter()
            .filter_map(|source| source.header.clone())
            .collect();
        // The rule's own two fields say a list of headers, each at the
        // default length. The request's id and the stripped suffixes take
        // the binding's block, and the rule then names no header.
        let by_rule = !key.strip_lineage
            && key
                .sources
                .iter()
                .all(|source| !source.rid && source.max_bytes.is_none());
        if !by_rule {
            within_a_block(&key, config)?;
        }
        let sticky_header = headers
            .first()
            .filter(|_| by_rule)
            .cloned()
            .unwrap_or_default();
        // The manual policy sweeps nothing where either of its two times is
        // zero: a key stays pinned for ever, which the rule says with its
        // largest number.
        let (ttl_s, sweep_s) = if max_idle_secs == 0 || eviction_interval_secs == 0 {
            (u32::MAX, u32::MAX)
        } else {
            (
                seconds("max_idle_secs", max_idle_secs)?,
                seconds("eviction_interval_secs", eviction_interval_secs)?,
            )
        };
        view_headers.clone_from(&headers);
        // A pair of presets names no rule: the manual preset's own is the
        // one the settings are written into.
        let preset = match binding.sessions.take() {
            Some(rule) => Some(rule),
            None => PolicySpec::preset("manual")?
                .bindings
                .pop()
                .and_then(|binding| binding.sessions),
        };
        binding.routing_key = (!by_rule).then_some(key);
        binding.sessions = Some(SessionsSpec {
            sticky_header,
            // One header is the sticky header alone, and the rule lists none.
            headers: if by_rule && headers.len() > 1 {
                headers
            } else {
                Vec::new()
            },
            ttl_s,
            sweep_s,
            assignment: match assignment_mode {
                ManualAssignmentMode::Random => Assignment::Random,
                ManualAssignmentMode::MinLoad => Assignment::MinLoad,
                ManualAssignmentMode::MinGroup => Assignment::MinGroup,
                ManualAssignmentMode::Delegate => Assignment::Delegate,
            },
            ..preset.ok_or(PolicyDocumentError::NoPart {
                policy: "manual",
                part: "session rule",
            })?
        });
        Ok(())
    }
}

/// The one list of sources the keyed sides read, or the two policies whose
/// lists differ: a binding names one list for every side. The sides are held
/// to the same session sources. Client sources are the hash ring's alone, and
/// no other side's stage reads their key, so the binding takes them from the
/// side that names them.
fn one_list(
    keys: Vec<(&'static str, RoutingKeySpec)>,
) -> Result<Option<RoutingKeySpec>, PolicyDocumentError> {
    let of_session = |source: &&KeySourceSpec| !source.client;
    let mut keys = keys.into_iter();
    let Some((first, mut key)) = keys.next() else {
        return Ok(None);
    };
    for (second, other) in keys {
        let same = other.strip_lineage == key.strip_lineage
            && (other.sources.iter().filter(of_session)).eq(key.sources.iter().filter(of_session));
        if !same {
            return Err(unsaid(
                "policy",
                format!("{first} and {second}"),
                "the two sides read two lists of key sources and a binding names one",
            ));
        }
        if other.sources.len() > key.sources.len() {
            key = other;
        }
    }
    Ok(Some(key))
}

/// A `routing_key` block holds eight sources at most, its client sources
/// not counted.
fn within_a_block(key: &RoutingKeySpec, config: &RouterConfig) -> Result<(), PolicyDocumentError> {
    if key.sources.iter().filter(|source| !source.client).count() > MAX_KEY_SOURCES {
        return Err(unsaid(
            "routing_key_override.headers",
            spelled(&config.routing_key_override.headers),
            "a binding names eight sources of its key at most",
        ));
    }
    Ok(())
}

/// The two headers the hash ring reads as a client's key, in its order.
const CLIENT_HEADERS: [&str; 2] = ["x-forwarded-for", "cookie"];

/// The sources a keyed policy reads a request's routing key from, in the
/// order it reads them, as a binding's `routing_key` block names them: an
/// empty list where the policy reads a key and no source is configured.
/// `None` for a policy that reads no key.
///
/// - Every keyed policy reads the configured headers first, lowercase and
///   each once.
/// - `manual` and `consistent_hashing` read the routing key header behind
///   them. Under the override they read the request's own id behind that,
///   and every source loses its turn and retry suffixes.
/// - `prefix_hash` reads the configured headers alone.
///
/// Two reads have no document yet and are refused by the setting's name.
/// The override places every other policy's keyed requests by the sticky
/// sessions' rule. And under the override the gateway reads the routing
/// key header as sent where no configured header names it, while a block
/// strips every source or none.
///
/// `consistent_hashing` also names its two client headers as client
/// sources: a key from them places a request that carries no session key,
/// and lifts no overload veto. A configured `x-forwarded-for` is named both
/// ways, as the gateway reads it: within the cap as a session's key, and
/// whole as a client's.
fn key_sources(
    policy: &PolicyConfig,
    config: &RouterConfig,
) -> Result<Option<RoutingKeySpec>, PolicyDocumentError> {
    let overridden = config.routing_key_override.enabled;
    let hint = match policy {
        PolicyConfig::Manual { .. } | PolicyConfig::ConsistentHashing => true,
        PolicyConfig::PrefixHash { .. } if !overridden => false,
        // Each kind is named: a new one builds once its answer is written.
        PolicyConfig::PrefixHash { .. }
        | PolicyConfig::Random
        | PolicyConfig::RoundRobin
        | PolicyConfig::Passthrough
        | PolicyConfig::CacheAware { .. }
        | PolicyConfig::PowerOfTwo { .. }
        | PolicyConfig::LeastLoad { .. }
        | PolicyConfig::Bucket { .. } => {
            return if overridden {
                Err(unsaid(
                    "routing_key_override.enabled",
                    true,
                    "the controller holds no recorded form of the sticky sessions' rule yet",
                ))
            } else {
                Ok(None)
            };
        }
    };
    let header = |name: &str, client| KeySourceSpec {
        header: Some(name.to_string()),
        client,
        ..KeySourceSpec::default()
    };
    let reads = |sources: &[KeySourceSpec], name: &str| {
        sources
            .iter()
            .any(|source| source.header.as_deref() == Some(name))
    };
    let mut sources: Vec<KeySourceSpec> = Vec::new();
    for name in &config.routing_key_override.headers {
        let name = name.to_ascii_lowercase();
        if !reads(&sources, &name) {
            sources.push(header(&name, false));
        }
    }
    if hint && !reads(&sources, MANUAL_SESSION_HEADER) {
        if overridden {
            return Err(unsaid(
                "routing_key_override.headers",
                spelled(&config.routing_key_override.headers),
                "under the override the routing key header is read as sent behind these headers, and a binding strips every source or none",
            ));
        }
        sources.push(header(MANUAL_SESSION_HEADER, false));
    }
    if overridden {
        sources.push(KeySourceSpec {
            rid: true,
            ..KeySourceSpec::default()
        });
    }
    if matches!(policy, PolicyConfig::ConsistentHashing) {
        sources.extend(CLIENT_HEADERS.map(|name| header(name, true)));
    }
    Ok(Some(RoutingKeySpec {
        sources,
        strip_lineage: overridden,
    }))
}

/// The binding's overload block, or none where the configuration stands at
/// the block's own defaults.
fn overload(config: &RouterConfig) -> Result<Option<OverloadSpec>, PolicyDocumentError> {
    // A threshold at none switches its signal off in the gateway. The
    // document reads an absent threshold as the controller's own number, so
    // it cannot say one signal off alone. Both off is the block off.
    let waiting = config.worker_overload_waiting_requests;
    let usage = config.worker_overload_token_usage;
    let enabled = config.worker_overload_protection && (waiting.is_some() || usage.is_some());
    if enabled {
        let one_off = "the document's overload block cannot switch one signal off alone";
        if waiting.is_none() {
            return Err(unsaid("worker_overload_waiting_requests", "null", one_off));
        }
        if usage.is_none() {
            return Err(unsaid("worker_overload_token_usage", "null", one_off));
        }
    }
    let written = OverloadSpec {
        enabled,
        waiting_requests: waiting
            .map(|waiting| narrow("worker_overload_waiting_requests", waiting))
            .transpose()?
            .and_then(|waiting| unless(waiting, OVERLOAD_WAITING_REQUESTS)),
        token_usage: usage.and_then(|usage| unless(usage as f32, OVERLOAD_TOKEN_USAGE)),
        shed_when_all_vetoed: config.worker_overload_shed,
    };
    Ok(unless(written, OverloadSpec::default()))
}

#[cfg(test)]
mod tests {
    use serde_json::json;
    use smg_policy_controller::PRESETS;

    use super::*;

    fn config(policy: PolicyConfig) -> RouterConfig {
        RouterConfig {
            policy,
            ..RouterConfig::default()
        }
    }

    /// [`super::at_flag_defaults`] for a preset of this build.
    fn at_flag_defaults(name: &str) -> PolicyConfig {
        super::at_flag_defaults(name).unwrap_or_else(|| {
            panic!("the controller has a preset this module converts no policy to: {name}")
        })
    }

    fn preset(name: &str) -> PolicySpec {
        PolicySpec::preset(name).expect("a preset of this build")
    }

    #[test]
    fn with_no_document_the_flags_document_is_in_force() {
        let config = config(PolicyConfig::RoundRobin);
        assert_eq!(config.policy_document_in_force(), Ok(preset("round_robin")));
    }

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
    fn a_given_document_that_the_flags_mean_too_is_in_force() {
        let document = preset("round_robin");
        let config = RouterConfig {
            policy_spec: Some(document.clone()),
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(config.policy_document_in_force(), Ok(document));
    }

    #[test]
    fn the_generation_numbers_the_document_and_is_no_part_of_the_comparison() {
        for (written, runs) in [(0, 1), (1, 1), (7, 7)] {
            let mut document = preset("round_robin");
            document.generation = written;
            let mut in_force = document.clone();
            in_force.generation = runs;
            for settings in [RouterConfig::default(), config(PolicyConfig::RoundRobin)] {
                let config = RouterConfig {
                    policy_spec: Some(document.clone()),
                    ..settings
                };
                assert_eq!(
                    config.policy_document_in_force().as_ref(),
                    Ok(&in_force),
                    "generation {written}"
                );
            }
        }
    }

    #[test]
    fn a_routing_flag_that_says_something_else_is_refused_with_both_values() {
        let mut document = preset("bucket");
        let profile = document.profiles.remove("bucket").expect("the profile");
        document.profiles.insert("power_of_two".to_owned(), profile);
        document.bindings[0].profile = Some("power_of_two".to_owned());
        let config = RouterConfig {
            policy_spec: Some(document),
            ..config(at_flag_defaults("power_of_two"))
        };
        let refusal = config
            .policy_document_in_force()
            .expect_err("different stages");
        assert_eq!(
            refusal.to_string(),
            r#"the routing flags and the policy document disagree at profiles.power_of_two.score[0]: the flags mean "least-load", the document says "bucket""#
        );
    }

    #[test]
    fn a_side_policy_that_says_something_else_is_refused_with_both_values() {
        let config = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: vec![],
                decode_urls: vec![],
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy: None,
            },
            policy_spec: Some(
                PolicySpec::pair_presets("random", "random").expect("paired presets"),
            ),
            ..RouterConfig::default()
        };
        assert_eq!(
            config.policy_document_in_force(),
            Err(PolicyDocumentError::Disagree {
                at: "profiles.prefill.pick".to_owned(),
                flags: r#""round-robin""#.to_owned(),
                document: r#""random""#.to_owned(),
            }),
        );
    }

    #[test]
    fn a_part_one_side_lacks_reads_nothing() {
        let document = edited(preset("round_robin"), |form| {
            form["bindings"][0]["shadow"] = json!("round_robin");
        });
        let mut config = RouterConfig {
            policy_spec: Some(document),
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(),
            Err(PolicyDocumentError::Disagree {
                at: "bindings[0].shadow".to_owned(),
                flags: "nothing".to_owned(),
                document: r#""round_robin""#.to_owned(),
            }),
        );

        let mut document = preset("round_robin");
        let binding = serde_json::to_string(&document.bindings[0]).expect("the binding serializes");
        document.bindings.clear();
        config.policy_spec = Some(document);
        assert_eq!(
            config.policy_document_in_force(),
            Err(PolicyDocumentError::Disagree {
                at: "bindings[0]".to_owned(),
                flags: binding,
                document: "nothing".to_owned(),
            }),
        );
    }

    #[test]
    fn a_kind_in_an_otherwise_equal_document_is_refused_by_name() {
        let document = edited(preset("round_robin"), |form| {
            form["kind"] = json!("PolicySpec");
        });
        let config = RouterConfig {
            policy_spec: Some(document),
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(),
            Err(PolicyDocumentError::Preset(SpecError::Kind(
                "PolicySpec".to_owned()
            )))
        );
    }

    /// A document with places of its JSON form written over.
    fn edited(document: PolicySpec, edit: impl FnOnce(&mut Value)) -> PolicySpec {
        let mut form = serde_json::to_value(document).expect("a document serializes");
        edit(&mut form);
        serde_json::from_value(form).expect("the edited document")
    }

    fn refusal(config: &RouterConfig) -> (&'static str, String) {
        match config.policy_document() {
            Err(PolicyDocumentError::Unsaid { setting, value, .. }) => (setting, value),
            other => panic!("expected a setting refused by name, got {other:?}"),
        }
    }

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

    /// A threshold at none switches its signal off in the gateway, and an
    /// absent threshold of the document is the controller's own number. One
    /// signal off alone is refused by its field's name, whatever the other
    /// is. Both off is the block switched off.
    #[test]
    fn an_overload_signal_switched_off_alone_is_refused_and_both_are_the_block_off() {
        let thresholds = |waiting: Option<usize>, usage: Option<f64>| {
            let mut config = config(at_flag_defaults("round_robin"));
            config.worker_overload_waiting_requests = waiting;
            config.worker_overload_token_usage = usage;
            config
        };
        let null = "null".to_string();
        for (waiting, usage, named) in [
            (None, Some(0.8), "worker_overload_waiting_requests"),
            (None, Some(0.5), "worker_overload_waiting_requests"),
            (Some(8), None, "worker_overload_token_usage"),
            (Some(32), None, "worker_overload_token_usage"),
        ] {
            assert_eq!(
                refusal(&thresholds(waiting, usage)),
                (named, null.clone()),
                "waiting {waiting:?}, usage {usage:?}"
            );
        }
        let document = thresholds(None, None)
            .policy_document()
            .expect("both signals off have a document");
        assert!(!document.bindings.is_empty());
        for binding in &document.bindings {
            let block = binding.overload.as_ref().expect("the block is written");
            assert!(!block.enabled, "both signals off is the block off");
            assert_eq!((block.waiting_requests, block.token_usage), (None, None));
        }
        // With the protection off the thresholds decide nothing: none is refused.
        let mut unprotected = thresholds(None, Some(0.8));
        unprotected.worker_overload_protection = false;
        assert!(unprotected.policy_document().is_ok());
    }

    /// The converter puts a written stage in the place of a preset's stage
    /// at the preset's weight, and carries no other parameter of it. So a
    /// preset's scorer that has parameters has a weight alone. A preset that
    /// gains another parameter fails here first, and the converter carries
    /// that parameter from then on. `consistent-hash` is the one stage the
    /// converter writes whole from the configuration.
    #[test]
    fn every_presets_scorer_with_parameters_has_a_weight_alone() {
        for name in PRESETS {
            let form = serde_json::to_value(preset(name)).expect("a document serializes");
            let profiles = form["profiles"].as_object().expect("the profiles");
            for (profile, stages) in profiles {
                for scorer in stages["score"].as_array().into_iter().flatten() {
                    let Some(with_parameters) = scorer.as_object() else {
                        continue;
                    };
                    for (stage, parameters) in with_parameters {
                        if stage == "consistent-hash" {
                            continue;
                        }
                        let named: Vec<&String> = parameters
                            .as_object()
                            .into_iter()
                            .flatten()
                            .map(|(name, _)| name)
                            .collect();
                        assert_eq!(named, ["weight"], "{name}: {profile}: {stage}");
                    }
                }
            }
        }
    }

    #[test]
    fn power_of_two_lands_on_the_form_of_its_metric() {
        for (load_metric, form) in [
            (
                PowerOfTwoLoadMetric::ExpectedWait,
                PowerOfTwoMetric::ExpectedWait,
            ),
            (PowerOfTwoLoadMetric::Requests, PowerOfTwoMetric::Requests),
            (
                PowerOfTwoLoadMetric::LeastRequests,
                PowerOfTwoMetric::LeastRequests,
            ),
            (
                PowerOfTwoLoadMetric::ResidentTokens {
                    request_tokens: 512,
                },
                PowerOfTwoMetric::ResidentTokens {
                    request_tokens: 512,
                },
            ),
        ] {
            assert_eq!(
                config(PolicyConfig::PowerOfTwo { load_metric }).policy_document(),
                Ok(PolicySpec::power_of_two(form).expect("a form of the preset")),
                "{load_metric:?} is the preset in that metric's form"
            );
        }
    }

    #[test]
    fn a_pair_mode_is_the_pair_of_presets_under_its_pairing_rule() {
        let pair = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: Some(PolicyConfig::RoundRobin),
                decode_policy: None,
            },
            pd_pairing_mode: PdPairingMode::Strict,
            ..config(at_flag_defaults("least_load"))
        };
        let mut expected =
            PolicySpec::pair_presets("round_robin", "least_load").expect("a pair of presets");
        expected.plans.pd_pair.pairing = Pairing::Strict;
        assert_eq!(
            pair.policy_document(),
            Ok(expected),
            "a side with no policy of its own takes the main one, and the pairing mode is the plan's"
        );
    }

    #[test]
    fn an_encode_side_with_no_policy_is_placed_by_consistent_hashing() {
        let encode_pair = RouterConfig {
            mode: RoutingMode::EncodePrefillDecode {
                encode_urls: Vec::new(),
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                encode_policy: None,
                prefill_policy: None,
                decode_policy: Some(PolicyConfig::Random),
            },
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            encode_pair.policy_document(),
            Ok(
                PolicySpec::epd_presets("consistent_hashing", "round_robin", "random")
                    .expect("the encode pair's presets")
            ),
        );
    }

    #[test]
    fn a_cache_aware_setting_lands_in_the_stage_that_reads_it() {
        let tuned = RouterConfig {
            policy: PolicyConfig::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                balance_prefix_margin_tokens: 1024,
                balance_uncached_credit: true,
                eviction_interval_secs: 60,
                max_tree_size: 16_777_216,
                block_size: 256,
                balance_token_usage_threshold: 0.3,
                overload_token_usage_threshold: 0.9,
                overlap_decay: 0.5,
                selection_temperature: 0.05,
                cache_index: CacheIndexKind::Hash,
                cache_ttl_secs: 300,
                cache_boundaries: vec![512, 2048],
                selection_policy: None,
                selection_accounting_ttl_ms: 200,
            },
            worker_warmup_secs: 0,
            worker_warmup_blocks: 2048,
            ..RouterConfig::default()
        };
        let expected = edited(preset("cache_aware"), |document| {
            document["view"] = json!({ "block_size": 256, "boundaries": [512, 2048] });
            document["index"] = json!({
                "*": { "kind": "hash", "ttl_s": 300, "max_size": 16_777_216, "sweep_s": 60 }
            });
            let profile = &mut document["profiles"]["cache_aware"];
            profile["score"][0] = json!({
                "warmup-slice": { "weight": 8.0, "secs": 0, "blocks": 2048, "miss_share": 0.5 }
            });
            profile["score"][1] = json!({ "prefix-affinity": {
                "weight": 4.0,
                "threshold": 0.5,
                "decay": 0.5,
                "accounting_ttl_ms": 200,
                "spill": {
                    "abs": 32, "rel": 1.1, "prefix_margin_tokens": 1024, "uncached_credit": true
                },
            }});
            profile["score"][2] = json!({ "kv-pressure": { "spread": 0.3, "ceiling": 0.9 } });
            profile["pick"] = json!({ "best-with-temperature": { "temperature": 0.05 } });
        });
        assert_eq!(tuned.policy_document(), Ok(expected));
    }

    #[test]
    fn the_settings_of_the_other_policies_land_in_their_stages() {
        let least_load = config(PolicyConfig::LeastLoad {
            kv_pressure_weight: 0.3,
            mean_prefill_tokens: 1024,
            default_throughput: 2000.0,
        });
        assert_eq!(
            least_load.policy_document(),
            Ok(edited(preset("least_load"), |document| {
                document["profiles"]["least_load"]["score"][0] =
                    json!({ "least-load": { "kv_pressure_weight": 0.3 } });
            })),
            "one setting off its default is the one parameter written"
        );
        let bucket = config(PolicyConfig::Bucket {
            balance_abs_threshold: 32,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 10,
        });
        assert_eq!(
            bucket.policy_document(),
            Ok(edited(preset("bucket"), |document| {
                document["profiles"]["bucket"]["score"][0] =
                    json!({ "bucket": { "abs_threshold": 32, "adjust_interval_s": 10 } });
            })),
        );
        let prefix_hash = config(PolicyConfig::PrefixHash {
            prefix_token_count: 512,
            load_factor: 1.5,
            balance_abs_threshold: 4,
            cache_boundaries: vec![256, 1024, 4096],
        });
        assert_eq!(
            prefix_hash.policy_document(),
            Ok(edited(preset("prefix_hash"), |document| {
                document["view"] = json!({ "boundaries": [256, 1024, 4096] });
                document["profiles"]["prefix_hash"]["score"][0] = json!({ "consistent-hash": {
                    "weight": 8.0,
                    "prefix_tokens": 512,
                    "bounded_load": { "factor": 1.5, "abs_margin": 4 },
                }});
            })),
        );
    }

    #[test]
    fn the_manual_policy_writes_the_bindings_session_rule() {
        let manual = PolicyConfig::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::MinGroup,
        };
        let rule = |document: &mut Value| {
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["ttl_s"] = json!(3600);
            sessions["sweep_s"] = json!(30);
            sessions["assignment"] = json!("min_group");
        };
        assert_eq!(
            config(manual.clone()).policy_document(),
            Ok(edited(preset("manual"), rule))
        );

        // Either time at zero sweeps nothing: a key stays pinned for ever.
        let pinned = config(PolicyConfig::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 0,
            assignment_mode: ManualAssignmentMode::Random,
        })
        .policy_document()
        .expect("a document");
        let for_ever = |document: &mut Value| {
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["ttl_s"] = json!(u32::MAX);
            sessions["sweep_s"] = json!(u32::MAX);
        };
        assert_eq!(pinned, edited(preset("manual"), for_ever));
        assert!(pinned.validate().is_ok(), "the controller takes the rule");

        // The sweep at zero beside an idle time: the gateway starts no
        // sweep, so the key stays pinned for ever too.
        let unswept = config(PolicyConfig::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: ManualAssignmentMode::Random,
        })
        .policy_document()
        .expect("a document");
        assert_eq!(unswept, edited(preset("manual"), for_ever));

        // A pair of manual sides: the pair of presets names no rule, and the
        // manual preset's own is the one written.
        let mut pair = PolicySpec::pair_presets("manual", "manual").expect("a pair of presets");
        pair.bindings[0].sessions = preset("manual").bindings[0].sessions.clone();
        let both = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(manual.clone())
        };
        assert_eq!(both.policy_document(), Ok(edited(pair, rule)));

        let one_side = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: None,
                decode_policy: Some(PolicyConfig::RoundRobin),
            },
            ..config(manual)
        };
        assert_eq!(
            refusal(&one_side),
            ("policy", "manual".to_string()),
            "one session rule serves the whole binding, so a manual side beside another policy is refused"
        );
    }

    #[test]
    fn the_headers_of_a_routing_key_land_where_the_policy_reads_its_key() {
        let keyed = |policy: &str, headers: &[&str]| {
            let mut keyed = config(at_flag_defaults(policy));
            keyed.routing_key_override.headers =
                headers.iter().map(|name| (*name).to_string()).collect();
            keyed
        };

        // The manual policy reads the configured headers, then the routing
        // key header. Its rule names them: the first is the sticky one, and
        // the rule lists them where there is more than one, each once and
        // lowercase.
        let both = |document: &mut Value| {
            document["view"]["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
            let sessions = &mut document["bindings"][0]["sessions"];
            sessions["sticky_header"] = json!("x-session-id");
            sessions["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
        };
        let listed = ["X-Session-Id", "x-smg-routing-key", "x-session-id"];
        assert_eq!(
            keyed("manual", &listed).policy_document(),
            Ok(edited(preset("manual"), both))
        );
        assert_eq!(
            keyed("manual", &["x-session-id"]).policy_document(),
            Ok(edited(preset("manual"), both)),
            "the routing key header is read behind a configured one that is not it"
        );
        assert_eq!(
            keyed("manual", &[]).policy_document(),
            Ok(preset("manual")),
            "with no header configured the manual policy still reads the routing key header"
        );

        // The prefix ring reads the configured headers alone, and none is no
        // block. The hash ring reads the routing key header behind them.
        let sources = |name: &str, sources: Value| {
            edited(preset(name), |document| {
                document["bindings"][0]["routing_key"] = json!({ "sources": sources });
            })
        };
        assert_eq!(
            keyed("prefix_hash", &["x-session-id", "X-Other-Key"]).policy_document(),
            Ok(sources(
                "prefix_hash",
                json!([{ "header": "x-session-id" }, { "header": "x-other-key" }])
            ))
        );
        assert_eq!(
            keyed("prefix_hash", &[]).policy_document(),
            Ok(edited(preset("prefix_hash"), |document| {
                document["bindings"][0]["routing_key"] = Value::Null;
            }))
        );
        assert_eq!(
            keyed("consistent_hashing", &["x-session-id"]).policy_document(),
            Ok(sources(
                "consistent_hashing",
                json!([
                    { "header": "x-session-id" },
                    { "header": "x-smg-routing-key" },
                    { "header": "x-forwarded-for", "client": true },
                    { "header": "cookie", "client": true }
                ])
            ))
        );

        // Eight sources bound a block. The manual policy's rule holds the
        // configured headers, bounded by the view: 256 with the routing key
        // header.
        let eight: Vec<String> = (0..8).map(|n| format!("x-key-{n}")).collect();
        let eight: Vec<&str> = eight.iter().map(String::as_str).collect();
        assert_eq!(
            refusal(&keyed("consistent_hashing", &eight)).0,
            "routing_key_override.headers"
        );
        assert!(keyed("manual", &eight).policy_document().is_ok());
        // The hash ring's two client sources are counted apart: seven
        // configured headers and the routing key header are its eight.
        let seven = keyed("consistent_hashing", &eight[..7]);
        let written = seven.policy_document().expect("eight sources");
        assert!(seven.hosted_policy(&written).is_ok());

        // The hash ring reads a configured `x-forwarded-for` both ways,
        // within the cap as a session's key and whole as a client's. The
        // document names it twice, and the controller's check takes it.
        let both_ways = keyed("consistent_hashing", &["x-forwarded-for"]);
        let document = sources(
            "consistent_hashing",
            json!([
                { "header": "x-forwarded-for" },
                { "header": "x-smg-routing-key" },
                { "header": "x-forwarded-for", "client": true },
                { "header": "cookie", "client": true }
            ]),
        );
        assert_eq!(both_ways.policy_document().as_ref(), Ok(&document));
        assert!(both_ways.hosted_policy(&document).is_ok());

        // No other policy reads a key, so its document names no source.
        let keyed_policies = ["manual", "prefix_hash", "consistent_hashing"];
        for name in PRESETS
            .iter()
            .filter(|name| !keyed_policies.contains(*name))
        {
            assert_eq!(
                keyed(name, &["x-session-id"]).policy_document(),
                Ok(preset(name)),
                "{name}"
            );
        }
    }

    #[test]
    fn the_override_adds_the_requests_id_where_a_policy_keeps_its_own_key() {
        let overridden = |policy: &str| {
            let mut overridden = config(at_flag_defaults(policy));
            overridden.routing_key_override.enabled = true;
            overridden
        };

        // The hash ring keeps its own placement under the override. Its
        // sources gain the request's id behind the headers, and every source
        // loses its turn and retry suffixes.
        assert_eq!(
            overridden("consistent_hashing").policy_document(),
            Ok(edited(preset("consistent_hashing"), |document| {
                document["bindings"][0]["routing_key"] = json!({
                    "sources": [
                        { "header": "x-smg-routing-key" },
                        { "rid": true },
                        { "header": "x-forwarded-for", "client": true },
                        { "header": "cookie", "client": true }
                    ],
                    "strip_lineage": true
                });
            }))
        );

        // The manual policy's rule says headers alone. The id and the strip
        // take the binding's block, and the rule then names no header.
        assert_eq!(
            overridden("manual").policy_document(),
            Ok(edited(preset("manual"), |document| {
                let binding = &mut document["bindings"][0];
                binding["routing_key"] = json!({
                    "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                    "strip_lineage": true
                });
                binding["sessions"]["sticky_header"] = json!("");
            }))
        );

        // The override places every other policy's keyed requests itself.
        for name in ["prefix_hash", "cache_aware", "round_robin"] {
            assert_eq!(
                refusal(&overridden(name)),
                ("routing_key_override.enabled", "true".to_string()),
                "{name}"
            );
        }

        // Behind other configured headers the gateway reads the routing key
        // header as sent, and a block strips every source or none.
        let mut behind = overridden("consistent_hashing");
        behind.routing_key_override.headers = vec!["x-session-id".to_string()];
        assert_eq!(
            refusal(&behind),
            (
                "routing_key_override.headers",
                r#"["x-session-id"]"#.to_string()
            )
        );

        // The encode side reads no key and the override never places it.
        let mut plan = RouterConfig {
            mode: RoutingMode::EncodePrefillDecode {
                encode_urls: Vec::new(),
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                encode_policy: Some(PolicyConfig::RoundRobin),
                prefill_policy: Some(PolicyConfig::ConsistentHashing),
                decode_policy: Some(PolicyConfig::ConsistentHashing),
            },
            ..config(PolicyConfig::RoundRobin)
        };
        plan.routing_key_override.enabled = true;
        let sources = plan.policy_document().map(|document| {
            document
                .bindings
                .first()
                .and_then(|binding| binding.routing_key.as_ref())
                .map(|key| key.sources.len())
        });
        assert_eq!(sources, Ok(Some(4)));
    }

    #[test]
    fn the_sides_of_a_plan_share_one_list_of_key_sources() {
        let pair = |prefill: &str, decode: &str| RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: Some(at_flag_defaults(prefill)),
                decode_policy: Some(at_flag_defaults(decode)),
            },
            pd_pairing_mode: PdPairingMode::Strict,
            ..config(PolicyConfig::RoundRobin)
        };

        // One keyed side names the binding's sources, whichever side it is.
        for (prefill, decode) in [
            ("consistent_hashing", "least_load"),
            ("least_load", "consistent_hashing"),
            ("prefix_hash", "prefix_hash"),
            ("prefix_hash", "consistent_hashing"),
        ] {
            let mut expected =
                PolicySpec::pair_presets(prefill, decode).expect("a pair of presets");
            expected.plans.pd_pair.pairing = Pairing::Strict;
            assert_eq!(
                pair(prefill, decode).policy_document(),
                Ok(expected),
                "{prefill} and {decode}"
            );
        }

        // Two sides that read two lists are refused by their policies: with
        // no header configured the prefix ring reads none, and the hash ring
        // still reads the routing key header.
        let mut bare = pair("prefix_hash", "consistent_hashing");
        bare.routing_key_override.headers = Vec::new();
        assert_eq!(
            refusal(&bare),
            ("policy", "prefix_hash and consistent_hashing".to_string())
        );

        // The manual policy's own refusal comes first.
        assert_eq!(
            refusal(&pair("manual", "consistent_hashing")),
            ("policy", "manual".to_string())
        );
    }

    #[test]
    fn the_overload_thresholds_land_on_the_binding_only_off_their_defaults() {
        let tightened = RouterConfig {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            worker_overload_shed: true,
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            tightened.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({
                    "waiting_requests": 32, "token_usage": 0.9, "shed_when_all_vetoed": true
                });
            })),
        );
        let off = RouterConfig {
            worker_overload_protection: false,
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            off.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({ "enabled": false });
            })),
        );
    }

    #[test]
    fn a_setting_no_document_says_is_refused_by_its_name() {
        let round_robin = || config(PolicyConfig::RoundRobin);
        let mut sticky = round_robin();
        sticky.routing_key_override.enabled = true;
        let mut selection = config(at_flag_defaults("cache_aware"));
        if let PolicyConfig::CacheAware {
            selection_policy, ..
        } = &mut selection.policy
        {
            *selection_policy = Some("another".to_string());
        }
        for (refused, setting, value) in [
            (sticky, "routing_key_override.enabled", "true"),
            (
                RouterConfig {
                    dp_minimum_tokens_scheduler: true,
                    ..round_robin()
                },
                "dp_minimum_tokens_scheduler",
                "true",
            ),
            (
                RouterConfig {
                    mode: RoutingMode::OpenAI {
                        worker_urls: Vec::new(),
                    },
                    ..round_robin()
                },
                "mode",
                "openai",
            ),
            (selection, "selection_policy", "another"),
        ] {
            assert_eq!(refusal(&refused), (setting, value.to_string()));
        }
    }

    #[test]
    fn two_sides_that_name_two_values_of_one_place_are_refused() {
        let sides = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: Some(PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864)),
                decode_policy: Some(PolicyConfig::cache_aware(0.3, 64, 1.5, 60, 67_108_864)),
            },
            ..config(PolicyConfig::RoundRobin)
        };
        let (setting, value) = refusal(&sides);
        assert_eq!(
            setting, "cache_index, cache_ttl_secs, max_tree_size, eviction_interval_secs",
            "the index settings stand once under a model, so two sweep intervals have no document: {value}"
        );
        // Each side's value reads as the configuration spells it.
        assert_eq!(
            value,
            r#"["tree",180,67108864,120] and ["tree",180,67108864,60]"#
        );

        // Two block sizes: the view holds one.
        let sized = |blocks: usize| {
            let mut side = PolicyConfig::cache_aware(0.3, 64, 1.5, 120, 67_108_864);
            if let PolicyConfig::CacheAware { block_size, .. } = &mut side {
                *block_size = blocks;
            }
            side
        };
        let blocks = RouterConfig {
            mode: RoutingMode::PrefillDecode {
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                prefill_policy: Some(sized(16)),
                decode_policy: Some(sized(32)),
            },
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            refusal(&blocks),
            ("block_size", "16 and 32".to_string()),
            "the view holds one block size, so two sides with two have no document"
        );
    }

    #[test]
    fn a_number_past_the_documents_range_is_refused() {
        let for_ever = config(PolicyConfig::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: u64::from(u32::MAX),
            assignment_mode: ManualAssignmentMode::Random,
        });
        assert_eq!(refusal(&for_ever).0, "max_idle_secs");
        let wide = config(PolicyConfig::PowerOfTwo {
            load_metric: PowerOfTwoLoadMetric::ResidentTokens {
                request_tokens: u64::from(u32::MAX) + 1,
            },
        });
        assert_eq!(refusal(&wide).0, "load_metric.request_tokens");
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
        assert!(matches!(
            given.policy_document_in_force(),
            Err(PolicyDocumentError::Unsaid {
                setting: "grouped_dp_sticky",
                ..
            })
        ));
    }

    #[test]
    fn a_balance_threshold_past_the_documents_width_lands_as_its_largest_number() {
        // A threshold no load reaches: the classic's way to switch a spill
        // off. The document's largest number is one too, and the controller
        // takes it.
        let never = usize::MAX;
        let bucket = config(PolicyConfig::Bucket {
            balance_abs_threshold: never,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        let prefix_hash = config(PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: never,
            cache_boundaries: Vec::new(),
        });
        let cache_aware = config(PolicyConfig::cache_aware(0.3, never, 1.5, 120, 67_108_864));
        for (config, at) in [
            (bucket, "/profiles/bucket/score/0/bucket/abs_threshold"),
            (
                prefix_hash,
                "/profiles/prefix_hash/score/0/consistent-hash/bounded_load/abs_margin",
            ),
            (
                cache_aware,
                "/profiles/cache_aware/score/1/prefix-affinity/spill/abs",
            ),
        ] {
            let document = config.policy_document().expect("a document");
            let written = serde_json::to_value(&document).expect("a written form");
            assert_eq!(written.pointer(at), Some(&json!(u32::MAX)), "{at}");
            assert_eq!(config.hosted_policy(&document).err(), None, "{at}");
        }
    }

    #[test]
    fn an_index_the_configuration_never_sweeps_is_never_trimmed() {
        // The classic trims its index at its sweep alone, and sweeps nothing
        // at an interval of zero: the configured size bounds nothing. The
        // document says so with its longest cadence and its largest size,
        // which no write of the controller's record meets.
        let swept_every = |secs| PolicyConfig::cache_aware(0.3, 64, 1.5, secs, 67_108_864);
        let unswept = config(swept_every(0));
        let document = unswept.policy_document().expect("a document");
        assert_eq!(
            document.index.get("*"),
            Some(&IndexSpec {
                max_size: u64::MAX,
                sweep_s: u32::MAX,
                ..IndexSpec::default()
            })
        );
        assert_eq!(unswept.hosted_policy(&document).err(), None);
        // A swept index keeps the size it is trimmed to.
        let swept = config(PolicyConfig::cache_aware(0.3, 64, 1.5, 60, 1024))
            .policy_document()
            .expect("a document");
        assert_eq!(
            swept.index.get("*"),
            Some(&IndexSpec {
                max_size: 1024,
                sweep_s: 60,
                ..IndexSpec::default()
            })
        );

        // Two sides are still compared as the configuration spells them.
        let sides = RouterConfig {
            mode: pair_mode(Some(swept_every(0)), Some(swept_every(60))),
            ..config(PolicyConfig::RoundRobin)
        };
        assert_eq!(
            refusal(&sides).1,
            r#"["tree",180,67108864,0] and ["tree",180,67108864,60]"#
        );
    }

    /// `config` as a gateway of many routers runs it.
    fn many_routers(config: RouterConfig) -> RouterConfig {
        RouterConfig {
            enable_igw: true,
            ..config
        }
    }

    fn pair_mode(prefill: Option<PolicyConfig>, decode: Option<PolicyConfig>) -> RoutingMode {
        RoutingMode::PrefillDecode {
            prefill_urls: Vec::new(),
            decode_urls: Vec::new(),
            prefill_policy: prefill,
            decode_policy: decode,
        }
    }

    #[test]
    fn the_default_document_of_many_routers_binds_pairs_ahead_of_single_workers() {
        let document = many_routers(RouterConfig::default())
            .policy_document()
            .expect("the default document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["prefill", "decode"] },
                    "plan": "pd-pair",
                    "profiles": { "decode": "decode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "random" },
            ]),
        );
        assert_eq!(
            document.profiles.keys().collect::<Vec<_>>(),
            ["decode", "prefill", "random"]
        );
    }

    #[test]
    fn many_routers_say_each_family_as_its_own_configuration_does() {
        for name in PRESETS.iter().copied().filter(|name| *name != "manual") {
            let regular = config(at_flag_defaults(name));
            let pair = RouterConfig {
                mode: pair_mode(None, None),
                ..regular.clone()
            };
            let single = regular.policy_document().expect("the regular document");
            let paired = pair.policy_document().expect("the pair document");
            let both = many_routers(regular.clone())
                .policy_document()
                .expect("the document of many routers");

            // The pair binding leads over its two phases. The single binding
            // stands behind it as the regular configuration writes it.
            let mut bindings = paired.bindings.clone();
            for binding in &mut bindings {
                binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
            }
            bindings.extend(single.bindings.clone());
            assert_eq!(both.bindings, bindings, "{name}");
            let mut profiles = paired.profiles.clone();
            profiles.extend(single.profiles.clone());
            assert_eq!(both.profiles, profiles, "{name}");
            assert_eq!(
                (&both.view, &both.index),
                (&single.view, &single.index),
                "{name}"
            );

            // A pair configuration with no side policy of its own says the
            // same document, and the controller takes it.
            assert_eq!(
                many_routers(pair).policy_document().as_ref(),
                Ok(&both),
                "{name}"
            );
            assert_eq!(regular.hosted_policy(&both).err(), None, "{name}");
        }
    }

    #[test]
    fn a_setting_of_the_binding_lands_on_both_bindings_of_many_routers() {
        let mut keyed = RouterConfig {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            worker_overload_shed: true,
            pd_pairing_mode: PdPairingMode::Strict,
            ..config(PolicyConfig::ConsistentHashing)
        };
        keyed.routing_key_override.headers = vec!["x-user".to_string()];
        let single = keyed.policy_document().expect("the regular document");
        let (overload, key) = single
            .bindings
            .first()
            .map(|binding| (binding.overload.clone(), binding.routing_key.clone()))
            .expect("the regular binding");
        assert!(overload.is_some() && key.is_some());

        let both = many_routers(keyed)
            .policy_document()
            .expect("the document of many routers");
        assert_eq!(both.bindings.len(), 2);
        for binding in &both.bindings {
            // One set of thresholds judges the pools of each family, and
            // each family reads a request's key from the same sources.
            assert_eq!(binding.overload, overload);
            assert_eq!(binding.routing_key, key);
        }
        // The pairing mode is the pair plan's under a regular configuration too.
        assert_eq!(both.plans.pd_pair.pairing, Pairing::Strict);
    }

    #[test]
    fn a_given_document_is_compared_with_the_document_of_many_routers() {
        let given = preset("round_robin");
        for mode in [RouterConfig::default().mode, pair_mode(None, None)] {
            // No routing setting is set: the given document is in force.
            let unset = many_routers(RouterConfig {
                mode: mode.clone(),
                policy_spec: Some(given.clone()),
                ..RouterConfig::default()
            });
            assert_eq!(unset.policy_document_in_force(), Ok(given.clone()));

            // A setting that says another document is refused, and one that
            // says the given document is not.
            let set = many_routers(RouterConfig {
                mode,
                ..config(PolicyConfig::RoundRobin)
            });
            let said = set.policy_document().expect("the flags' document");
            let other = RouterConfig {
                policy_spec: Some(given.clone()),
                ..set.clone()
            };
            assert!(matches!(
                other.policy_document_in_force(),
                Err(PolicyDocumentError::Disagree { .. })
            ));
            let same = RouterConfig {
                policy_spec: Some(said.clone()),
                ..set
            };
            assert_eq!(same.policy_document_in_force(), Ok(said));
        }
    }

    #[test]
    fn many_routers_plan_one_family_where_the_other_has_no_document() {
        // The manual policy runs on no side of a pair.
        let manual = config(at_flag_defaults("manual"));
        assert!(manual.policy_document().is_ok());
        assert_eq!(
            many_routers(manual.clone()).policy_document(),
            manual.policy_document()
        );

        // The encode mode's document is its own.
        let encode = RouterConfig {
            mode: RoutingMode::EncodePrefillDecode {
                encode_urls: Vec::new(),
                prefill_urls: Vec::new(),
                decode_urls: Vec::new(),
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(PolicyConfig::RoundRobin)
        };
        assert!(encode.policy_document().is_ok());
        assert_eq!(
            many_routers(encode.clone()).policy_document(),
            encode.policy_document()
        );

        // Under the override the main policy's keyed requests have no
        // document yet, while the hash ring on both sides of the pair has
        // one. The pair configuration's document then plans pairs alone.
        let mut keyed = RouterConfig {
            mode: pair_mode(
                Some(PolicyConfig::ConsistentHashing),
                Some(PolicyConfig::ConsistentHashing),
            ),
            ..config(PolicyConfig::RoundRobin)
        };
        keyed.routing_key_override.enabled = true;
        keyed.routing_key_override.headers = vec![MANUAL_SESSION_HEADER.to_string()];
        let pairs_alone = keyed.policy_document();
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(keyed.clone()).policy_document(), pairs_alone);
        // The same main policy alone is refused by the setting's name.
        keyed.mode = RouterConfig::default().mode;
        assert_eq!(
            refusal(&many_routers(keyed)),
            ("routing_key_override.enabled", "true".to_string())
        );
    }

    /// A pair configuration with round robin on both sides, under `main`.
    fn round_robin_pairs_under(main: PolicyConfig) -> RouterConfig {
        RouterConfig {
            mode: pair_mode(
                Some(PolicyConfig::RoundRobin),
                Some(PolicyConfig::RoundRobin),
            ),
            ..config(main)
        }
    }

    /// The cache-aware policy at its flags' defaults, over blocks of `blocks`.
    fn cache_aware_over(blocks: usize) -> PolicyConfig {
        let mut policy = at_flag_defaults("cache_aware");
        if let PolicyConfig::CacheAware { block_size, .. } = &mut policy {
            *block_size = blocks;
        }
        policy
    }

    #[test]
    fn many_routers_write_the_view_and_index_of_a_main_policy_no_side_runs() {
        // Round robin places both sides of a pair and names no view and no
        // index. A tuned cache-aware policy places single workers: what it
        // shares with every side lands in the document all the same.
        let mut main = PolicyConfig::cache_aware(0.3, 64, 1.5, 60, 16_777_216);
        if let PolicyConfig::CacheAware { block_size, .. } = &mut main {
            *block_size = 256;
        }
        let single = config(main.clone())
            .policy_document()
            .expect("the regular document");
        assert!(single.view.block_size.is_some() && !single.index.is_empty());
        let pairs = round_robin_pairs_under(main);
        let paired = pairs.policy_document().expect("the pair document");
        let both = many_routers(pairs.clone())
            .policy_document()
            .expect("the document of many routers");

        assert_eq!(both.bindings.len(), 2);
        assert_eq!(both.bindings.last(), single.bindings.first());
        assert_eq!(
            both.profiles.get("cache_aware"),
            single.profiles.get("cache_aware")
        );
        for side in ["prefill", "decode"] {
            assert_eq!(both.profiles.get(side), paired.profiles.get(side), "{side}");
        }
        assert_eq!((&both.view, &both.index), (&single.view, &single.index));
        assert_eq!(pairs.hosted_policy(&both).err(), None);
    }

    #[test]
    fn many_routers_keep_the_manual_rule_on_the_single_binding_alone() {
        // The manual policy places single workers while round robin places
        // both sides of a pair. The session rule and the view's header are
        // the single binding's, and the pair binding holds no rule.
        let manual = at_flag_defaults("manual");
        let single = config(manual.clone())
            .policy_document()
            .expect("the regular document");
        let pairs = round_robin_pairs_under(manual);
        let both = many_routers(pairs.clone())
            .policy_document()
            .expect("the document of many routers");

        assert_eq!(both.bindings.len(), 2);
        let pair = both.bindings.first().expect("the pair binding");
        assert_eq!((&pair.sessions, &pair.routing_key), (&None, &None));
        let rule = both.bindings.last().expect("the single binding");
        assert!(rule.sessions.is_some());
        assert_eq!(Some(rule), single.bindings.first());
        assert_eq!(both.view, single.view);
        assert!(!both.view.headers.is_empty());
        assert_eq!(pairs.hosted_policy(&both).err(), None);
    }

    #[test]
    fn many_routers_plan_pairs_alone_where_the_main_policy_names_another_shared_value() {
        // The view holds one block size. The sides agree on theirs and the
        // main policy names another, so single workers have no document and
        // the pair configuration plans its own family alone.
        let clash = RouterConfig {
            mode: pair_mode(Some(cache_aware_over(16)), Some(cache_aware_over(16))),
            ..config(cache_aware_over(32))
        };
        let pairs_alone = clash.policy_document();
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(clash).policy_document(), pairs_alone);
    }

    const LABELLED: &str = "qwen3";

    fn labelled(models: &[(&str, &'static str)]) -> Vec<(String, &'static str)> {
        models
            .iter()
            .map(|&(model, name)| (model.to_owned(), name))
            .collect()
    }

    /// The labelled document of `config`, and the labels it left out, each
    /// with the setting its refusal names.
    fn bound(
        config: &RouterConfig,
        models: &[(&str, &'static str)],
    ) -> (PolicySpec, Vec<(String, String)>) {
        let (document, left) = config
            .policy_document_labelled(&labelled(models))
            .expect("the settings' document");
        let left = left
            .into_iter()
            .map(|(model, why)| (model, why.to_string()))
            .collect();
        (document, left)
    }

    /// One model whose first worker's label names another policy than the
    /// main one. Its binding stands ahead of the wildcard and is the
    /// binding of that policy's own document but for the model it matches,
    /// its profile is that document's, and nothing else of the settings'
    /// document moves.
    #[test]
    fn a_labelled_model_is_bound_ahead_of_the_wildcard_under_its_policys_preset() {
        let config = config(PolicyConfig::RoundRobin);
        let settled = config.policy_document().expect("the settings' document");
        let (document, left) = bound(&config, &[(LABELLED, "consistent_hashing")]);
        assert_eq!(left, []);
        let own = self::config(PolicyConfig::ConsistentHashing)
            .policy_document()
            .expect("the labelled policy's own document");
        let mut of_model = own.bindings[0].clone();
        assert!(of_model.routing_key.is_some(), "the ring reads a key");
        of_model.matcher.model = LABELLED.to_owned();
        assert_eq!(document.bindings, [of_model, settled.bindings[0].clone()]);
        assert_eq!(
            document.profiles["consistent_hashing"],
            own.profiles["consistent_hashing"]
        );
        let mut rest = document.clone();
        rest.bindings.remove(0);
        rest.profiles.remove("consistent_hashing");
        assert_eq!(rest, settled);
        config
            .hosted_policy(&document)
            .expect("the labelled document validates");
    }

    /// The labelled bindings stand in the order the models are handed, and
    /// two models of one label run one profile.
    #[test]
    fn labelled_models_stand_in_their_order_and_share_a_policys_profile() {
        let config = config(PolicyConfig::RoundRobin);
        let (document, left) = bound(
            &config,
            &[
                ("a", "consistent_hashing"),
                ("b", "least_load"),
                ("c", "consistent_hashing"),
            ],
        );
        assert_eq!(left, []);
        let matched: Vec<_> = document
            .bindings
            .iter()
            .map(|binding| (binding.matcher.model.as_str(), binding.profile.as_deref()))
            .collect();
        assert_eq!(
            matched,
            [
                ("a", Some("consistent_hashing")),
                ("b", Some("least_load")),
                ("c", Some("consistent_hashing")),
                ("*", Some("round_robin")),
            ]
        );
        assert_eq!(document.profiles.len(), 3);
        assert_eq!(
            document.profiles["least_load"],
            preset("least_load").profiles["least_load"]
        );
        config
            .hosted_policy(&document)
            .expect("the labelled document validates");
    }

    /// No label, and a label that names the main policy, write the
    /// settings' document: the wildcard's profile is that policy already.
    #[test]
    fn a_label_naming_the_main_policy_binds_nothing() {
        let config = config(at_flag_defaults("least_load"));
        let settled = config.policy_document().expect("the settings' document");
        assert_eq!(bound(&config, &[]), (settled.clone(), Vec::new()));
        assert_eq!(
            bound(&config, &[(LABELLED, "least_load")]),
            (settled, Vec::new())
        );
    }

    /// A label the document cannot say is left out with why, and the
    /// labels beside it stand: a policy that names another value where the
    /// document holds one for every side, a name no preset has, and a
    /// model's id a binding reads as a pattern.
    #[test]
    fn a_label_the_document_cannot_say_is_left_out_with_why() {
        // The main policy names two boundaries, and a cache-aware policy at
        // its preset's numbers names none.
        let config = config(PolicyConfig::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: 10,
            cache_boundaries: vec![64, 512],
        });
        let (document, left) = bound(
            &config,
            &[
                ("a", "cache_aware"),
                ("b", "round_robin"),
                ("c", "no_such_policy"),
                ("llama*", "round_robin"),
                ("a*b", "round_robin"),
                ("", "round_robin"),
            ],
        );
        let models: Vec<_> = document
            .bindings
            .iter()
            .map(|binding| binding.matcher.model.as_str())
            .collect();
        assert_eq!(models, ["b", "*"]);
        let pattern = "worker label policy = round_robin: a binding matches no model of this id \
                       alone: it reads the id as a pattern";
        assert_eq!(
            left,
            [
                (
                    "a".to_owned(),
                    "cache_boundaries = [] and [64,512]: two sides name two values and the \
                     document holds one for every side"
                        .to_owned()
                ),
                (
                    "c".to_owned(),
                    "worker label policy = no_such_policy: no preset has this name".to_owned()
                ),
                ("llama*".to_owned(), pattern.to_owned()),
                ("a*b".to_owned(), pattern.to_owned()),
                (String::new(), pattern.to_owned()),
            ]
        );
    }

    /// A label whose binding the door would refuse is left out with the
    /// door's reason, and the label behind it binds. Here the tenant
    /// header, which the gateway denies a document's view, is the header a
    /// consistent hashing profile keys by.
    #[test]
    fn a_label_the_door_would_refuse_is_left_out_and_the_next_one_binds() {
        let mut config = config(PolicyConfig::RoundRobin);
        config.tenant_resolution.tenant_header_name = "x-smg-routing-key".to_owned();
        let (document, left) = bound(&config, &[("a", "consistent_hashing"), ("b", "least_load")]);
        let models: Vec<_> = document
            .bindings
            .iter()
            .map(|binding| binding.matcher.model.as_str())
            .collect();
        assert_eq!(models, ["b", "*"]);
        let [(model, why)] = left.as_slice() else {
            panic!("one label is left out: {left:?}");
        };
        assert_eq!(model, "a");
        assert!(why.contains("is on the deny-list"), "{why}");
    }

    /// A document of pairs alone plans no single workers, so it binds no
    /// label. A gateway of many routers binds a labelled model behind its
    /// pairs and ahead of every other single worker's request.
    #[test]
    fn a_labelled_binding_stands_behind_the_pairs_and_only_beside_single_workers() {
        let pairs = RouterConfig {
            mode: pair_mode(None, None),
            ..config(PolicyConfig::RoundRobin)
        };
        let settled = pairs.policy_document().expect("the pair document");
        assert_eq!(
            bound(&pairs, &[(LABELLED, "consistent_hashing")]),
            (
                settled,
                vec![(
                    LABELLED.to_owned(),
                    "worker label policy = consistent_hashing: the document plans no single \
                     workers, and a side's policy is the mode's"
                        .to_owned()
                )]
            )
        );
        let many = many_routers(config(PolicyConfig::RoundRobin));
        let (document, left) = bound(&many, &[(LABELLED, "consistent_hashing")]);
        assert_eq!(left, []);
        let planned: Vec<_> = document
            .bindings
            .iter()
            .map(|binding| (binding.matcher.model.as_str(), binding.plan))
            .collect();
        assert_eq!(
            planned,
            [("*", Some(PlanName::PdPair)), (LABELLED, None), ("*", None),]
        );
        many.hosted_policy(&document)
            .expect("the labelled document validates");
    }
}
