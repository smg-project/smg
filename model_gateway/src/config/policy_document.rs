//! The routing configuration as a policy document:
//! [`RouterConfig::policy_document`], one function from the configuration to
//! the policy controller's [`PolicySpec`].
//!
//! # Why this exists
//!
//! The policy controller decides by a document, not by flags. This function
//! writes the document that today's routing settings mean, so a gateway can
//! hand its routing to the controller and keep the routing its flags asked
//! for. Nothing reads the document yet: no request path calls this module.
//!
//! # Where a setting lands
//!
//! - A policy name lands on the controller's preset of that name. A
//!   prefill-decode mode lands on the pair plan, with one profile a side.
//! - A policy's own setting lands in the stage of the preset that reads it,
//!   and only where it differs from the number that stage runs at when its
//!   document names none. The controller names those numbers and this
//!   module reads them, so a configuration at them is the preset itself.
//! - The overload thresholds land on the binding, the pairing mode on the
//!   pair plan, the index settings under every model, and the block size
//!   and the cache boundaries in the view.
//! - The manual policy's settings land in the binding's session rule, with
//!   the headers its key is read from.
//!
//! # What is refused
//!
//! A setting that decides a placement and has no place in the document is
//! refused by its name ([`PolicyDocumentError::Unsaid`]). It is never left
//! out in silence.
//!
//! # What is not read
//!
//! What the gateway keeps for itself under the controller is no part of the
//! document: the pools and their addresses, discovery, the probes and the
//! load poll, retries and the breaker, the prefill queue and the decode
//! rooms. The headers a routing key is read from are the gateway's too: it
//! reads the key and states it as a fact of the request. One place names
//! them, the session rule the manual policy places by. The copy of the
//! cache boundaries at the top of the configuration decides nothing either:
//! each policy reads its own.

use serde::Serialize;
use smg_policy_controller::{
    affinity::{
        Gates, DEFAULT_THRESHOLD, DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY,
        DEFAULT_WARMUP_SECS, DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
    },
    bucket,
    builtins::{DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT},
    spec::{
        BestWithTemperatureParams, BoundedLoadParams, BucketParams, ConsistentHashParams,
        IndexKind, IndexSpec, KvPressureParams, LeastLoadParams, OverloadSpec, Pairing, PickStage,
        PrefixAffinityParams, ProfileSpec, ScoreSpec, ScoreStage, SpillParams, WarmupSliceParams,
        OVERLOAD_TOKEN_USAGE, OVERLOAD_WAITING_REQUESTS,
    },
    AppliedSpec, Assignment, PolicySpec, PowerOfTwoMetric, SessionsSpec, SpecError,
};

use super::{
    CacheIndexKind, GroupedDpSticky, ManualAssignmentMode, PdPairingMode, PolicyConfig,
    PowerOfTwoLoadMetric, RouterConfig, RoutingMode,
};
use crate::policies::cost::DEFAULT_POLICY as DEFAULT_SELECTION_POLICY;

/// The encode side's policy where the configuration names none: encode
/// placement follows an item's content, not the request's main policy.
static ENCODE_DEFAULT: PolicyConfig = PolicyConfig::ConsistentHashing;

/// Why a routing configuration has no policy document.
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
    /// The controller knows no preset of a policy's name.
    #[error(transparent)]
    Preset(#[from] SpecError),
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
        let (mut document, sides) = presets(self)?;
        let mut shared = Shared::default();
        for &(profile, policy) in &sides {
            let profile =
                document
                    .profiles
                    .get_mut(profile)
                    .ok_or_else(|| PolicyDocumentError::NoPart {
                        policy: policy.name(),
                        part: "profile",
                    })?;
            write_policy(policy, self, profile, &mut shared)?;
        }
        shared.write(self, sides.len(), &mut document)?;
        let binding = document
            .bindings
            .first_mut()
            .ok_or_else(|| PolicyDocumentError::NoPart {
                policy: self.policy.name(),
                part: "binding",
            })?;
        binding.overload = overload(self)?;
        Ok(document)
    }

    /// The generation `document` validates into where this gateway hosts the
    /// controller: the controller's own check, with the gateway's tenant
    /// header denied to the view. The one check of `--print-policy-spec` and
    /// of the host, so the two never part.
    pub fn hosted_policy(&self, document: &PolicySpec) -> Result<AppliedSpec, SpecError> {
        let tenant_header = self.tenant_resolution.tenant_header_name.as_str();
        AppliedSpec::validate_with(document, &[tenant_header])
    }
}

/// The settings that decide a placement today and that no document says.
fn refuse_unsaid(config: &RouterConfig) -> Result<(), PolicyDocumentError> {
    if config.routing_key_override.enabled {
        return Err(unsaid(
            "routing_key_override.enabled",
            true,
            "the controller holds no recorded form of the sticky sessions' rule yet",
        ));
    }
    match config.grouped_dp_sticky {
        GroupedDpSticky::Off => {}
        GroupedDpSticky::Hash => {
            return Err(unsaid(
                "grouped_dp_sticky",
                GroupedDpSticky::Hash.as_str(),
                "the document has no rule that pins a key to one engine of a grouped worker",
            ));
        }
    }
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

/// The presets the mode's policies name, as one document, with its
/// profiles: one named for the policy under the regular mode, one a side
/// under the pair and the encode modes. The pools of a mode are the
/// gateway's and are not read.
fn presets(config: &RouterConfig) -> Result<(PolicySpec, Vec<Side<'_>>), PolicyDocumentError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (prefill, decode) = (mode.get_prefill_policy(main), mode.get_decode_policy(main));
    let (mut document, sides) = match mode {
        RoutingMode::Regular { worker_urls: _ } => {
            (PolicySpec::preset(main.name())?, vec![(main.name(), main)])
        }
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
                vec![("encode", encode), ("prefill", prefill), ("decode", decode)],
            )
        }
        RoutingMode::OpenAI { worker_urls: _ } => return Err(provider_mode("openai")),
        RoutingMode::Anthropic { worker_urls: _ } => return Err(provider_mode("anthropic")),
        RoutingMode::Gemini { worker_urls: _ } => return Err(provider_mode("gemini")),
    };
    // A document of several sides runs the pair: the pairing mode is its rule.
    if sides.len() > 1 {
        document.plans.pd_pair.pairing = match config.pd_pairing_mode {
            PdPairingMode::Off => Pairing::Off,
            PdPairingMode::Lenient => Pairing::Lenient,
            PdPairingMode::Strict => Pairing::Strict,
        };
    }
    Ok((document, sides))
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
                    narrow("balance_abs_threshold", *balance_abs_threshold)?,
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
            shared.manual_sides += 1;
            share(
                &mut shared.manual,
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
                abs_margin: Some(narrow("balance_abs_threshold", *balance_abs_threshold)?),
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
                abs: unless(
                    narrow("balance_abs_threshold", *balance_abs_threshold)?,
                    gates.abs,
                ),
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
    /// The manual policy's sweep interval, idle time and assignment.
    manual: Option<(u64, u64, ManualAssignmentMode)>,
    manual_sides: usize,
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
    /// Writes what the sides named into a document of `sides` profiles.
    fn write(
        self,
        config: &RouterConfig,
        sides: usize,
        document: &mut PolicySpec,
    ) -> Result<(), PolicyDocumentError> {
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
        let index = self
            .index
            .map(|(kind, ttl_s, max_size, sweep_s)| IndexSpec {
                kind: match kind {
                    CacheIndexKind::Tree => IndexKind::Tree,
                    CacheIndexKind::Hash => IndexKind::Hash,
                },
                ttl_s,
                max_size,
                sweep_s,
            });
        if let Some(index) = index.and_then(|index| unless(index, IndexSpec::default())) {
            document.index.insert("*".to_string(), index);
        }
        let Some((eviction_interval_secs, max_idle_secs, assignment_mode)) = self.manual else {
            return Ok(());
        };
        if self.manual_sides != sides {
            return Err(unsaid(
                "policy",
                "manual",
                "the manual policy places by the binding's one session rule, so every side of a plan is manual or none is",
            ));
        }
        let (sticky_header, headers) = key_headers(config)?;
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
        document.view.headers.clone_from(&headers);
        let binding = document
            .bindings
            .first_mut()
            .ok_or(PolicyDocumentError::NoPart {
                policy: "manual",
                part: "binding",
            })?;
        // A pair of presets names no rule: the manual preset's own is the
        // one the settings are written into.
        let preset = match binding.sessions.take() {
            Some(rule) => Some(rule),
            None => PolicySpec::preset("manual")?
                .bindings
                .pop()
                .and_then(|binding| binding.sessions),
        };
        binding.sessions = Some(SessionsSpec {
            sticky_header,
            // One header is the sticky header alone, and the rule lists none.
            headers: if headers.len() > 1 {
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

/// The headers a routing key is read from, as a session rule names them:
/// the first one, then all of them in the order they are tried, lowercase
/// and each once.
fn key_headers(config: &RouterConfig) -> Result<(String, Vec<String>), PolicyDocumentError> {
    let mut headers: Vec<String> = Vec::new();
    for name in &config.routing_key_override.headers {
        let name = name.to_ascii_lowercase();
        if !headers.contains(&name) {
            headers.push(name);
        }
    }
    match headers.first().cloned() {
        Some(first) => Ok((first, headers)),
        None => Err(unsaid(
            "routing_key_override.headers",
            "[]",
            "the manual policy's session rule names the header its key is read from",
        )),
    }
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
    use serde_json::{json, Value};
    use smg_policy_controller::PRESETS;

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
    fn the_headers_of_a_routing_key_land_in_the_manual_session_rule_alone() {
        let keyed = |policy: &str, headers: &[&str]| {
            let mut keyed = config(at_flag_defaults(policy));
            keyed.routing_key_override.headers =
                headers.iter().map(|name| (*name).to_string()).collect();
            keyed
        };

        // The first header is the sticky one. The rule lists them where
        // there is more than one, each once and lowercase.
        let listed = ["X-Session-Id", "x-smg-routing-key", "x-session-id"];
        assert_eq!(
            keyed("manual", &listed).policy_document(),
            Ok(edited(preset("manual"), |document| {
                document["view"]["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
                let sessions = &mut document["bindings"][0]["sessions"];
                sessions["sticky_header"] = json!("x-session-id");
                sessions["headers"] = json!(["x-session-id", "x-smg-routing-key"]);
            }))
        );
        assert_eq!(
            keyed("manual", &["x-session-id"]).policy_document(),
            Ok(edited(preset("manual"), |document| {
                document["view"]["headers"] = json!(["x-session-id"]);
                document["bindings"][0]["sessions"]["sticky_header"] = json!("x-session-id");
            }))
        );
        assert_eq!(
            refusal(&keyed("manual", &[])),
            ("routing_key_override.headers", "[]".to_string()),
            "a session rule with no header to read its key from is no document"
        );

        // No other policy's document names a header: the gateway reads the
        // key and states it.
        for name in PRESETS.iter().filter(|name| **name != "manual") {
            assert_eq!(
                keyed(name, &["x-session-id"]).policy_document(),
                Ok(preset(name)),
                "{name}"
            );
        }
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
                    grouped_dp_sticky: GroupedDpSticky::Hash,
                    ..round_robin()
                },
                "grouped_dp_sticky",
                "hash",
            ),
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
}
