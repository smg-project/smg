//! The translation of a gateway's routing settings into their policy
//! document: the presets the mode's policies name ([`presets`]) and the
//! configuration's settings written over them ([`written`]), a side at a
//! time, with what the sides of a document and of a binding share.

use std::collections::btree_map::Entry;

use serde::Serialize;

use super::{
    unsupported, ConnectionModeSetting, Labelled, LoadMetricSetting, ModeSettings, PolicySettings,
    RoutingSettings, RoutingSettingsError,
};
use crate::{
    document::spec::{
        Assignment, BestWithTemperatureParams, BindingSpec, BoundedLoadParams, BucketParams,
        ConsistentHashParams, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
        LeastLoadParams, OverloadSpec, PickStage, PolicySpec, PrefixAffinityParams, ProfileSpec,
        Recovery, RoutingKeySpec, RulePlaced, ScoreSpec, ScoreStage, SessionsSpec, SpecError,
        SpillParams, WarmupSliceParams, MANUAL_SESSION_HEADER, MAX_HEADERS, MAX_KEY_SOURCES,
    },
    engine::view::Phase,
    policies::{
        bucket,
        cache_aware::{
            Gates, DEFAULT_THRESHOLD, DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY,
            DEFAULT_WARMUP_SECS, DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
        },
        consistent_hashing::target_source,
        power_of_two::PowerOfTwoMetric,
    },
    stages::scorers::{
        DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
    },
};

impl ModeSettings {
    /// The prefill side's policy: its own, or the main one.
    fn prefill_policy<'a>(&'a self, main: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::PrefillDecode { prefill_policy, .. }
            | Self::EncodePrefillDecode { prefill_policy, .. } => {
                prefill_policy.as_ref().unwrap_or(main)
            }
            Self::Regular | Self::Provider(_) => main,
        }
    }

    /// The decode side's policy: its own, or the main one.
    fn decode_policy<'a>(&'a self, main: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::PrefillDecode { decode_policy, .. }
            | Self::EncodePrefillDecode { decode_policy, .. } => {
                decode_policy.as_ref().unwrap_or(main)
            }
            Self::Regular | Self::Provider(_) => main,
        }
    }

    /// The encode side's policy: its own, or `unnamed`. The main policy is
    /// never the encode side's.
    fn encode_policy<'a>(&'a self, unnamed: &'a PolicySettings) -> &'a PolicySettings {
        match self {
            Self::EncodePrefillDecode { encode_policy, .. } => {
                encode_policy.as_ref().unwrap_or(unnamed)
            }
            Self::Regular | Self::PrefillDecode { .. } | Self::Provider(_) => unnamed,
        }
    }
}

/// The encode side's policy where the configuration names none: encode
/// placement follows an item's content, not the request's main policy.
static ENCODE_DEFAULT: PolicySettings = PolicySettings::ConsistentHashing;

/// The encode side's profile, by its name in a document of three sides.
const ENCODE_SIDE: &str = "encode";

/// A profile of the document and the policy that decides it.
type Side<'a> = (&'static str, &'a PolicySettings);

/// A document and, binding by binding, the sides its plan runs.
type Planned<'a> = (PolicySpec, Vec<Vec<Side<'a>>>);

/// The presets the mode's policies name, as a document of one binding, with
/// its profiles: one named for the policy under the regular mode, one a side
/// under the pair and the encode modes. A provider mode of a multi-router
/// gateway reads as the regular mode: the document says the policy of its
/// self-hosted workers.
///
/// A pair mode's document binds two families ([`both_families`]), and the
/// encode mode's does in a gateway of many routers. A gateway of many
/// routers plans its three sides ahead of both ([`three_sided`]). The mode's
/// arm here is then reached only where the main policy's single binding has
/// no document, and plans the mode's own family alone.
pub(super) fn presets(config: &RoutingSettings) -> Result<Planned<'_>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (prefill, decode) = (mode.prefill_policy(main), mode.decode_policy(main));
    let regular = || -> Result<_, RoutingSettingsError> {
        Ok((PolicySpec::preset(main.name())?, vec![(main.name(), main)]))
    };
    let (document, sides) = match mode {
        ModeSettings::Regular => regular()?,
        ModeSettings::PrefillDecode {
            prefill_policy: _,
            decode_policy: _,
        } => (
            PolicySpec::pair_presets(prefill.name(), decode.name())?,
            vec![("prefill", prefill), ("decode", decode)],
        ),
        ModeSettings::EncodePrefillDecode {
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => {
            let encode = mode.encode_policy(&ENCODE_DEFAULT);
            (
                PolicySpec::epd_presets(encode.name(), prefill.name(), decode.name())?,
                vec![
                    (ENCODE_SIDE, encode),
                    ("prefill", prefill),
                    ("decode", decode),
                ],
            )
        }
        ModeSettings::Provider(_) => match config.provider_alone() {
            Some(provider) => return Err(provider_mode(provider)),
            None => regular()?,
        },
    };
    Ok((document, vec![sides]))
}

/// The document that binds pairs and single requests side by side: the pair
/// binding first, over the prefill and decode phases, and the single binding
/// behind it for every other request. Two configurations say it. A gateway
/// of many routers serves pairs and single workers. A gateway in the pair
/// mode, of many routers or of one, places one single request beside its
/// pairs over HTTP: a token count takes one prefill worker, and the
/// gateway's own routing places it by the main policy. Each family runs
/// under the policies the gateway's own routing gives it. The sides of a
/// pair run under the mode's side policies, which are the main policy in a
/// regular configuration. Single requests run under the main policy.
///
/// Over gRPC or ZMQ the pair mode counts tokens with no worker, as the
/// encode mode does. A gateway of one router then places no single request
/// and its document plans the pair alone: a binding no request reaches
/// would still add the main policy's needs to the document's. A gateway of
/// many routers serves single workers however it reaches them, in the
/// encode mode too: there the first binding is the mode's own, its three
/// sides over their three phases.
///
/// `None` where the document plans one family: a regular configuration of
/// one router, the encode mode of one router, a pair mode of one router
/// over gRPC or ZMQ, and a regular configuration under the manual policy.
/// Its picker leaves the placement to the binding's session rule, and this
/// module does not plan its pairs beside its single requests yet: a pair
/// mode whose two sides are manual is planned, under the one rule. A
/// gateway of one router in the encode mode counts tokens with no worker,
/// so its document binds no single request.
///
/// A gateway of many routers outside the encode mode plans its three sides
/// ahead of this plan ([`three_sided`]), and this one stands where that
/// plan has no document.
pub(super) fn both_families(
    config: &RoutingSettings,
) -> Result<Option<Planned<'_>>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (both, encode) = match mode {
        ModeSettings::Regular | ModeSettings::Provider(_) => (
            config.enable_igw && !matches!(main, PolicySettings::Manual { .. }),
            None,
        ),
        ModeSettings::PrefillDecode {
            prefill_policy: _,
            decode_policy: _,
        } => (
            config.enable_igw || config.connection_mode == ConnectionModeSetting::Http,
            None,
        ),
        ModeSettings::EncodePrefillDecode {
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => (config.enable_igw, Some(mode.encode_policy(&ENCODE_DEFAULT))),
    };
    if !both {
        return Ok(None);
    }
    families(config, encode).map(Some)
}

/// The document of a gateway of many routers outside the encode mode, with
/// an encode side: the three sides under one binding over their three
/// phases, and the single binding behind it. Such a gateway mounts the
/// three-sided router in every mode, and a model with an encode pool goes
/// to it. Its encode side runs under the encode side's default policy, and
/// its pair's two sides under the mode's side policies, as the pair plan's
/// do ([`both_families`]): the pair's two profiles are that plan's.
///
/// `None` for a gateway of one router, which mounts that router under the
/// encode mode alone, and for the encode mode, whose plan of two families
/// is this one already. A setting with no document under three sides
/// leaves the plan where it is written, as the manual policy on the pair's
/// two sides does, a regular configuration's main policy among them.
pub(super) fn three_sided(
    config: &RoutingSettings,
) -> Result<Option<Planned<'_>>, RoutingSettingsError> {
    let encode_mode = matches!(config.mode, ModeSettings::EncodePrefillDecode { .. });
    if !config.enable_igw || encode_mode {
        return Ok(None);
    }
    families(config, Some(&ENCODE_DEFAULT)).map(Some)
}

/// The plan of two families: the pair's sides under one binding over their
/// phases, with `encode` as a third side ahead of them where one is named,
/// and the main policy's single binding behind it.
fn families<'a>(
    config: &'a RoutingSettings,
    encode: Option<&'a PolicySettings>,
) -> Result<Planned<'a>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let (prefill, decode) = (mode.prefill_policy(main), mode.decode_policy(main));
    let mut sides = vec![("prefill", prefill), ("decode", decode)];
    let mut phases = vec![Phase::Prefill, Phase::Decode];
    let mut document = match encode {
        Some(encode) => {
            sides.insert(0, (ENCODE_SIDE, encode));
            phases.insert(0, Phase::Encode);
            PolicySpec::epd_presets(encode.name(), prefill.name(), decode.name())?
        }
        None => PolicySpec::pair_presets(prefill.name(), decode.name())?,
    };
    for binding in &mut document.bindings {
        binding.matcher.phase.clone_from(&phases);
    }
    let single = PolicySpec::preset(main.name())?;
    document.profiles.extend(single.profiles);
    document.bindings.extend(single.bindings);
    Ok((document, vec![sides, vec![(main.name(), main)]]))
}

/// Whether `document` was written from `plan`. The settings try their plans
/// in turn and write the first that stands. A plan of two families binds two
/// bindings and the mode's own plan one, and the two plans of two families
/// lead with two plans' names: the count of bindings and the first one's
/// plan say which plan a document came from.
pub(super) fn written_from((planned, bindings): &Planned<'_>, document: &PolicySpec) -> bool {
    let leads = |spec: &PolicySpec| spec.bindings.first().map(|binding| binding.plan);
    bindings.len() == document.bindings.len() && leads(planned) == leads(document)
}

/// Writes the configuration's settings over a planned document: each side's
/// policy into its profile, what the sides of a binding share into that
/// binding, and what every side shares into the places the document holds
/// once. The overload block lands on every binding, so one switch and one
/// shed word stand for the pools of each family.
pub(super) fn written<'a>(
    config: &'a RoutingSettings,
    planned: Planned<'a>,
) -> Result<PolicySpec, RoutingSettingsError> {
    written_with(config, planned, &[])
}

/// A label's refusal, from the builder's own for its side or its binding: a
/// setting the document has no place for under the label's policy. A
/// parting of the presets and this module is no label's, and passes as it
/// is: a host leaves a refused label out and calls again, which would
/// leave the parting unsaid.
pub(super) fn label_refused(
    model: &str,
    policy: &PolicySettings,
    refusal: RoutingSettingsError,
) -> RoutingSettingsError {
    match refusal {
        RoutingSettingsError::UnsupportedSetting { .. }
        | RoutingSettingsError::Disagreement { .. } => RoutingSettingsError::Label {
            model: model.to_owned(),
            policy: policy.name().to_owned(),
            reason: refusal.to_string(),
        },
        // Each kind is named: a new one builds once its answer is written.
        RoutingSettingsError::NoStage { .. }
        | RoutingSettingsError::NoPart { .. }
        | RoutingSettingsError::Document(_)
        | RoutingSettingsError::Label { .. } => refusal,
    }
}

/// One label's side of the document, written as a side of the settings is:
/// its policy's preset as a profile of the policy's name, the policy's
/// settings in that profile and in the places the document's sides share,
/// and what its one binding holds. Models that name one policy share its
/// profile, which a second write leaves as the first left it.
fn label_side<'a>(
    policy: &'a PolicySettings,
    config: &RoutingSettings,
    document: &mut PolicySpec,
    shared: &mut Shared<'a>,
) -> Result<Bound, RoutingSettingsError> {
    let name = policy.name();
    let profile = match document.profiles.entry(name.to_owned()) {
        Entry::Occupied(held) => held.into_mut(),
        Entry::Vacant(place) => {
            let preset = PolicySpec::preset(name)?.profiles.remove(name);
            place.insert(preset.ok_or(RoutingSettingsError::NoPart {
                policy: name,
                part: "profile",
            })?)
        }
    };
    write_policy(policy, config, profile, shared)?;
    if let Some(sources) = key_sources(policy, config)? {
        shared.bound.keys.push((name, sources));
    }
    shared.bound.overridden += usize::from(override_places(policy, config));
    Ok(std::mem::take(&mut shared.bound))
}

/// The binding of one label's model: its policy's preset binding, for that
/// model alone, with what its side holds written as a plan of one side. The
/// headers its side lists in the view join the view's list behind the ones
/// that stand there, each once. The list holds a bounded count, which the
/// document's own check holds for every model at once: a label whose
/// headers would pass it is refused here, in that check's words, so it
/// costs no other model its document.
fn label_binding(
    model: &str,
    policy: &PolicySettings,
    side: Bound,
    config: &RoutingSettings,
    view_headers: &mut Vec<String>,
) -> Result<BindingSpec, RoutingSettingsError> {
    let name = policy.name();
    let mut binding =
        PolicySpec::preset(name)?
            .bindings
            .pop()
            .ok_or(RoutingSettingsError::NoPart {
                policy: name,
                part: "binding",
            })?;
    model.clone_into(&mut binding.matcher.model);
    // A side lists each of its headers once. The ones the view holds stay
    // where they stand, and a list the settings' own sides took past the
    // bound is theirs: a label that adds no header is not refused for it.
    let mut headers = Vec::new();
    side.write(config, 1, &mut binding, &mut headers)?;
    headers.retain(|header| !view_headers.contains(header));
    let count = view_headers.len() + headers.len();
    if !headers.is_empty() && count > MAX_HEADERS {
        let bound = SpecError::TooManyHeaders {
            count,
            max: MAX_HEADERS,
        };
        return Err(RoutingSettingsError::Label {
            model: model.to_owned(),
            policy: policy.name().to_owned(),
            reason: bound.to_string(),
        });
    }
    view_headers.extend(headers);
    Ok(binding)
}

/// [`written`] with the labels' sides: each label is one more side of one,
/// written by its policy as the main policy's side is, and bound for its
/// model alone right ahead of the binding of every model's single requests.
/// A plan with no such binding binds no label. The settings' own sides are
/// written first and their bindings first, so a label changes none of them
/// but the view's headers, which it adds to. A label's ring keeps its
/// blocks under every gateway: no document of a label's stood before it.
pub(super) fn written_with<'a>(
    config: &'a RoutingSettings,
    (mut document, bindings): Planned<'a>,
    labels: &'a [Labelled<'a>],
) -> Result<PolicySpec, RoutingSettingsError> {
    // A plan of several sides runs the pair: the pairing mode is its rule.
    if bindings.iter().any(|sides| sides.len() > 1) {
        document.plans.pd_pair.pairing = config.pd_pairing_mode;
    }
    let mut shared = Shared::default();
    let mut bound = Vec::with_capacity(bindings.len());
    for sides in &bindings {
        for &(side, policy) in sides {
            let profile =
                document
                    .profiles
                    .get_mut(side)
                    .ok_or_else(|| RoutingSettingsError::NoPart {
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
                if override_places(policy, config) {
                    shared.bound.overridden += 1;
                } else {
                    shared.bound.keyed_itself = Some(side);
                }
            }
        }
        bound.push((sides.len(), std::mem::take(&mut shared.bound)));
    }
    // A pair mode of one router binds its single request for one side
    // route, a token count: that binding leaves the pairs' keys where they
    // were. Many routers serve each family as a fleet of its own, and keep
    // the document's one size for both, as they did. The encode mode of
    // many routers had its own document alone: its rings keep their blocks
    // too, so the single binding costs it no document that was taken.
    let encode_mode = matches!(config.mode, ModeSettings::EncodePrefillDecode { .. });
    if !config.enable_igw || encode_mode {
        rings_keep_their_blocks(shared.block_size, &bound)?;
    }
    // The binding of every model's single requests: the last plan of one
    // side. A label's side shares the document's one view with the sides
    // above, and a label's ring keeps its blocks beside the size they
    // name, in a gateway of many routers too: a label's ring is a new one.
    // A label names the size a silent ring counts in, so it moves no ring
    // of theirs and none of a label before it: its own side is the one to
    // read.
    let single = bindings.iter().rposition(|sides| sides.len() == 1);
    let mut sides_of_labels = Vec::with_capacity(labels.len());
    if single.is_some() {
        for (model, policy) in labels {
            let refused = |refusal: RoutingSettingsError| label_refused(model, policy, refusal);
            let side = label_side(policy, config, &mut document, &mut shared).map_err(refused)?;
            let side = (1, side);
            rings_keep_their_blocks(shared.block_size, std::slice::from_ref(&side))
                .map_err(refused)?;
            sides_of_labels.push(side);
        }
    }
    shared.write(&mut document)?;
    if document.bindings.is_empty() || document.bindings.len() != bound.len() {
        return Err(RoutingSettingsError::NoPart {
            policy: config.policy.name(),
            part: "binding",
        });
    }
    for (binding, (sides, bound)) in document.bindings.iter_mut().zip(bound) {
        bound.write(config, sides, binding, &mut document.view.headers)?;
    }
    if let Some(single) = single {
        for (at, ((model, policy), (_, side))) in labels.iter().zip(sides_of_labels).enumerate() {
            let binding = label_binding(model, policy, side, config, &mut document.view.headers)
                .map_err(|refusal| label_refused(model, policy, refusal))?;
            document.bindings.insert(single + at, binding);
        }
    }
    let overload = overload(config);
    for binding in &mut document.bindings {
        binding.overload.clone_from(&overload);
    }
    Ok(document)
}

/// A prefix ring keeps the blocks of its own binding's document, in a
/// gateway of one router and in the encode mode of many routers. The view
/// holds one block size for every profile. A ring names none: its key ends
/// with the last whole block at or under its depth, counted in the view's
/// size. So a size another binding's policy names would end the key at
/// another token, or leave it no whole block, which the controller's check
/// refuses. Such settings are an unsupported setting of the other family,
/// and the document plans the mode's own family alone. Sides of one binding
/// that name a size beside a ring say one document together, as they did.
fn rings_keep_their_blocks(
    block_size: Option<usize>,
    bound: &[(usize, Bound)],
) -> Result<(), RoutingSettingsError> {
    let Some(named) = block_size else {
        return Ok(());
    };
    let silent = usize::try_from(Gates::default().block_size).unwrap_or(usize::MAX);
    let own = |binding: &Bound| binding.block_size.unwrap_or(silent);
    match bound
        .iter()
        .find(|(_, binding)| binding.rings && own(binding) != named)
    {
        Some((_, binding)) => Err(unsupported(
            "block_size",
            format!("{} and {named}", own(binding)),
            "a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own",
        )),
        None => Ok(()),
    }
}

fn provider_mode(mode: &'static str) -> RoutingSettingsError {
    unsupported(
        "mode",
        mode,
        "a provider mode places its requests by no routing policy",
    )
}

/// Writes one side's policy over its preset: its settings into the stages of
/// its profile, and into the places the document's sides share.
fn write_policy<'a>(
    policy: &'a PolicySettings,
    config: &RoutingSettings,
    profile: &mut ProfileSpec,
    shared: &mut Shared<'a>,
) -> Result<(), RoutingSettingsError> {
    let name = policy.name();
    match policy {
        PolicySettings::Random
        | PolicySettings::RoundRobin
        | PolicySettings::Passthrough
        | PolicySettings::ConsistentHashing => Ok(()),
        PolicySettings::PowerOfTwo { load_metric } => {
            let metric = match *load_metric {
                LoadMetricSetting::ExpectedWait => PowerOfTwoMetric::ExpectedWait,
                LoadMetricSetting::Requests => PowerOfTwoMetric::Requests,
                LoadMetricSetting::LeastRequests => PowerOfTwoMetric::LeastRequests,
                LoadMetricSetting::ResidentTokens { request_tokens } => {
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
                .ok_or(RoutingSettingsError::NoPart {
                    policy: name,
                    part: "profile",
                })?;
            (profile.score, profile.pick) = (form.score, form.pick);
            Ok(())
        }
        PolicySettings::LeastLoad {
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
        PolicySettings::Bucket {
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
        PolicySettings::Manual {
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
        PolicySettings::PrefixHash {
            prefix_token_count,
            load_factor,
            balance_abs_threshold,
            cache_boundaries,
        } => {
            // The preset names the ring's three numbers itself, so they are
            // written whatever they are.
            let prefix_tokens = Some(narrow("prefix_token_count", *prefix_token_count)?);
            shared.bound.rings = true;
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
        PolicySettings::CacheAware {
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
            if let Some(named) = selection_policy.as_deref() {
                return Err(unsupported(
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
            shared.bound.block_size = Some(*block_size);
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
) -> Result<(), RoutingSettingsError> {
    let missing = RoutingSettingsError::NoStage { policy, stage };
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
fn narrow<T, U>(setting: &'static str, value: U) -> Result<T, RoutingSettingsError>
where
    T: TryFrom<U>,
    U: Copy + ToString,
{
    T::try_from(value).map_err(|_| unsupported(setting, value, "past the document's range"))
}

/// A balance threshold in the document's width. One past the width lands as
/// the width's largest number: no load reaches either, so the stage that
/// reads it decides the same under both.
fn threshold(value: usize) -> u32 {
    u32::try_from(value).unwrap_or(u32::MAX)
}

/// Seconds of a session rule in the document's width. The rule reads its
/// largest value as for ever, so a setting that large has no place in it.
fn seconds(setting: &'static str, secs: u64) -> Result<u32, RoutingSettingsError> {
    narrow(setting, secs).and_then(|narrowed: u32| {
        unless(narrowed, u32::MAX).ok_or_else(|| {
            unsupported(
                setting,
                secs,
                "the document reads this many seconds as for ever",
            )
        })
    })
}

/// A sticky map's idle time and sweep interval as a session rule's `ttl_s`
/// and `sweep_s`, each with the name of its setting. The map sweeps nothing
/// where either time is zero: a key the table holds stays pinned for ever,
/// which the rule says with its largest number.
fn pin_times(
    (idle_setting, max_idle_secs): (&'static str, u64),
    (sweep_setting, eviction_interval_secs): (&'static str, u64),
) -> Result<(u32, u32), RoutingSettingsError> {
    if max_idle_secs == 0 || eviction_interval_secs == 0 {
        return Ok((u32::MAX, u32::MAX));
    }
    Ok((
        seconds(idle_setting, max_idle_secs)?,
        seconds(sweep_setting, eviction_interval_secs)?,
    ))
}

/// What the sides of a document share. The document holds each once, so the
/// first side to name one sets it, and a side that names another value is
/// refused.
#[derive(Default)]
struct Shared<'a> {
    block_size: Option<usize>,
    boundaries: Option<&'a [usize]>,
    /// The index's kind, entry lifetime, size and sweep interval.
    index: Option<(IndexKind, u32, u64, u32)>,
    /// What the sides of the binding being written share.
    bound: Bound,
}

/// What the sides of one binding share. The binding holds each once.
#[derive(Default)]
struct Bound {
    /// The manual policy's sweep interval, idle time and assignment.
    manual: Option<(u64, u64, Assignment)>,
    manual_sides: usize,
    /// The sides whose keyed requests the sticky override places
    /// ([`override_places`]).
    overridden: usize,
    /// A side the override does not place. With the override on, that
    /// side's policy reads the routing key itself.
    keyed_itself: Option<&'static str>,
    /// The sources each keyed side reads a request's key from, under the
    /// side's policy.
    keys: Vec<(&'static str, RoutingKeySpec)>,
    /// The block size a side of the binding names.
    block_size: Option<usize>,
    /// A side of the binding places by a prefix ring, which counts its key
    /// in the view's blocks and names no size ([`rings_keep_their_blocks`]).
    rings: bool,
}

/// A value as the configuration spells it.
fn spelled<T: Serialize>(value: &T) -> String {
    serde_json::to_string(value).unwrap_or_else(|unspelled| unspelled.to_string())
}

fn share<T: PartialEq + Serialize>(
    held: &mut Option<T>,
    value: T,
    setting: &'static str,
) -> Result<(), RoutingSettingsError> {
    match held {
        Some(held) if *held != value => Err(unsupported(
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
    fn write(self, document: &mut PolicySpec) -> Result<(), RoutingSettingsError> {
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
            // The gateway's own policy trims its index at its sweep alone,
            // and sweeps nothing at an interval of zero: that index is never
            // trimmed.
            // The document's cadence and size both start at one, so its
            // longest cadence and its largest size say it. No sweep falls
            // inside a process's life, and no write meets the bound.
            let (max_size, sweep_s) = if sweep_s == 0 {
                (u64::MAX, u32::MAX)
            } else {
                (max_size, sweep_s)
            };
            IndexSpec {
                kind,
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
    /// profiles. The headers the manual policy reads are the view's too,
    /// and so are the headers of a key the sticky override places.
    fn write(
        self,
        config: &RoutingSettings,
        sides: usize,
        binding: &mut BindingSpec,
        view_headers: &mut Vec<String>,
    ) -> Result<(), RoutingSettingsError> {
        let Some((eviction_interval_secs, max_idle_secs, assignment_mode)) = self.manual else {
            // A binding names its key's sources once: what its keyed sides
            // read, or nothing, over whatever its presets wrote.
            let key = one_list(self.keys)?.filter(|key| !key.sources.is_empty());
            if let Some(key) = &key {
                within_a_block(key, config)?;
            }
            if self.overridden > 0 {
                if let Some(reason) = override_refused(sides, self.keyed_itself) {
                    return Err(unsupported("routing_key_override.enabled", true, reason));
                }
                binding.sessions = Some(override_rule(config)?);
                // The rule's cap counts a key's requests on a worker by
                // the key's text, which a header hands where the view
                // lists it.
                *view_headers = key
                    .iter()
                    .flat_map(|key| &key.sources)
                    .filter_map(|source| source.header.clone())
                    .collect();
            }
            binding.routing_key = key;
            return Ok(());
        };
        if self.manual_sides != sides {
            return Err(unsupported(
                "policy",
                "manual",
                "the manual policy places by the binding's one session rule, so every side of a plan is manual or none is",
            ));
        }
        let key = one_list(self.keys)?.ok_or(RoutingSettingsError::NoPart {
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
        let (ttl_s, sweep_s) = pin_times(
            ("max_idle_secs", max_idle_secs),
            ("eviction_interval_secs", eviction_interval_secs),
        )?;
        view_headers.clone_from(&headers);
        // A pair of presets names no rule: the manual preset's own is the
        // one the settings are written into. It keeps the preset's word of
        // a request the rule places, the profile's own: the gateway's
        // `manual` policy places a keyed request itself
        // (`model_gateway/src/policies/registry.rs:701`), and no stage
        // counts under its picker.
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
            assignment: assignment_mode,
            ..preset.ok_or(RoutingSettingsError::NoPart {
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
/// side that names them, and its target source with them.
fn one_list(
    keys: Vec<(&'static str, RoutingKeySpec)>,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let of_session = |source: &&KeySourceSpec| !source.client && !source.target;
    let mut keys = keys.into_iter();
    let Some((first, mut key)) = keys.next() else {
        return Ok(None);
    };
    for (second, other) in keys {
        let same = other.strip_lineage == key.strip_lineage
            && (other.sources.iter().filter(of_session)).eq(key.sources.iter().filter(of_session));
        if !same {
            return Err(unsupported(
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
/// and its target source not counted.
fn within_a_block(
    key: &RoutingKeySpec,
    config: &RoutingSettings,
) -> Result<(), RoutingSettingsError> {
    let of_session = |source: &&KeySourceSpec| !source.client && !source.target;
    if key.sources.iter().filter(of_session).count() > MAX_KEY_SOURCES {
        return Err(unsupported(
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
/// - `prefix_hash` reads the configured headers alone. Under the override
///   it reads the request's own id behind them, each source without its
///   turn and retry suffixes, and the binding's session rule pins that key
///   ([`override_places`]).
/// - `random`, `round_robin`, `power_of_two`, `least_load`, `passthrough`,
///   `bucket` and `cache_aware` read no key themselves. Under the override
///   they read one ([`override_places`]):
///   the configured headers, then the request's own id, each without its
///   turn and retry suffixes. The binding's session rule pins that key
///   ([`override_rule`]).
///
/// Two reads have no document yet and are refused by the setting's name.
/// The override's rule is written for a gateway of one router alone. And
/// under the override the gateway reads the routing key header as sent
/// where no configured header names it, while a block strips every source
/// or none.
///
/// `consistent_hashing` also names its two client headers as client
/// sources: a key from them places a request that carries no session key,
/// and lifts no overload veto. A configured `x-forwarded-for` is named both
/// ways, as the gateway reads it: within the cap as a session's key, and
/// whole as a client's.
///
/// `consistent_hashing` alone names its target header, as its target
/// source ([`target_source`]): a request that
/// carries it goes to the worker its number reaches, and no other policy's
/// request reads it.
fn key_sources(
    policy: &PolicySettings,
    config: &RoutingSettings,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let overridden = config.routing_key_override.enabled;
    let hint = match policy {
        PolicySettings::Manual { .. } | PolicySettings::ConsistentHashing => true,
        PolicySettings::PrefixHash { .. } if !overridden => false,
        _ if override_places(policy, config) => {
            if config.enable_igw {
                return Err(unsupported(
                    "routing_key_override.enabled",
                    true,
                    "the sticky sessions' rule has no document in a gateway of many routers yet",
                ));
            }
            false
        }
        // Each kind is named: a new one builds once its answer is written.
        // The override places every kind listed here (`override_places`),
        // so one lands here with the override off, where it reads no key.
        // `prefix_hash` never lands here: it reads the configured headers
        // above. A kind the override did not place would bind no session
        // rule and no key under it, so it is refused by the setting's name.
        PolicySettings::PrefixHash { .. }
        | PolicySettings::Random
        | PolicySettings::RoundRobin
        | PolicySettings::Passthrough
        | PolicySettings::CacheAware { .. }
        | PolicySettings::PowerOfTwo { .. }
        | PolicySettings::LeastLoad { .. }
        | PolicySettings::Bucket { .. } => {
            return if overridden {
                Err(unsupported(
                    "routing_key_override.enabled",
                    true,
                    "the sticky sessions' rule has no document under this policy yet",
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
            return Err(unsupported(
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
    if matches!(policy, PolicySettings::ConsistentHashing) {
        sources.extend(CLIENT_HEADERS.map(|name| header(name, true)));
        sources.push(target_source());
    }
    Ok(Some(RoutingKeySpec {
        sources,
        strip_lineage: overridden,
    }))
}

/// Whether the binding's session rule places a keyed request of `policy`
/// ([`override_rule`]): the sticky override is on, and the gateway's
/// sticky map places the policy's keyed requests. It does for every
/// policy but `manual` and `consistent_hashing`, which place by the key
/// themselves. Of the others `prefix_hash` alone has a stage that reads a
/// key, and every other places a request by nothing its key hands it.
///
/// - `prefix_hash` hashes the key a header handed in its prompt's place,
///   and no key the request's own id handed: it places that request by
///   its prompt (`model_gateway/src/policies/prefix_hash.rs:384`,
///   `:387-412`). The gateway asks the policy for a key's first request
///   and for a key at its pin's cap
///   (`model_gateway/src/policies/registry.rs:832`, `:803`), and hands
///   it the header's key without its turn and retry suffixes (`:646`).
///   This crate's ring does the same in a gateway's own process, where
///   the one call says that a key is the id's
///   ([`RequestFacts::session_key_from_rid`](crate::engine::RequestFacts::session_key_from_rid)).
///   A gateway's frame says it to a served controller. The frame of a
///   host older than that field says nothing, and its controller's ring
///   places the request by the key its id handed.
/// - `random` draws.
/// - `round_robin` takes turns. The gateway asks one policy object for a
///   request with no key, for a key's first request and for a key at its
///   pin's cap (`model_gateway/src/policies/registry.rs:722`, `:832`,
///   `:803`), so the three take the turns of one rotation, and a request
///   that follows its pin takes none (`:794-797`). The rotation is the
///   list's own, by row, and where the row's worker takes no request, the
///   rotation of the workers that do
///   (`model_gateway/src/policies/round_robin.rs:142-168`). This crate's
///   stage turns the same two for the three.
/// - `power_of_two` and `least_load` compare workers by load. Both score
///   a worker by its report and by the requests the policy placed on it
///   since. A request the rule placed is none of those: the rule keeps it
///   from the profile ([`override_rule`]), so in a gateway's own process
///   no stage hears its dispatch (`lease_events::stages_hear`), as the
///   gateway's policy hears of no request its sticky map placed
///   (`model_gateway/src/policies/registry.rs:794-797`). Both read such a
///   request in the count of requests the gateway holds on the worker. A
///   controller keeps the rule's word on the lease it files, and tells no
///   stage that dispatch either.
/// - `passthrough` sends every request to the first worker that takes one.
/// - `bucket` draws for a single request. Its stage places a prefill
///   decision by the sizes it booked and draws for every other. The
///   gateway's policy holds a bucket of sizes for the prefill workers of a
///   pair alone, and draws among single workers
///   (`model_gateway/src/policies/registry.rs:1716-1727`,
///   `model_gateway/src/policies/bucket.rs:186-191`). It books a
///   request's size where it is asked for the prefill (`:178-181`), and
///   it is not asked behind a pin (`registry.rs:794-797`). The stage books
///   a decision it hears issued, and it hears none the rule placed alone
///   ([`override_rule`]).
/// - `cache_aware` reads who was sent a prompt. The gateway's policy writes
///   a prompt where it placed the request, and it is not asked behind a pin
///   its sticky map followed nor for a key the map's own assignment placed
///   (`model_gateway/src/policies/registry.rs:794-797`, `:814`, `:840`).
///   In a gateway's own process the record of sent prefixes, or a token
///   tree the host lends, takes a prompt behind a request the profile's
///   stages place, as under the policy alone, and behind none the rule
///   places alone: the rule keeps those from the profile
///   ([`override_rule`]), and the record's writes read that word
///   (`lease_events::stages_hear`). A controller's decision is this
///   engine's, so its record follows the word too.
fn override_places(policy: &PolicySettings, config: &RoutingSettings) -> bool {
    let placed = matches!(
        policy,
        PolicySettings::PrefixHash { .. }
            | PolicySettings::Random
            | PolicySettings::RoundRobin
            | PolicySettings::PowerOfTwo { .. }
            | PolicySettings::LeastLoad { .. }
            | PolicySettings::Passthrough
            | PolicySettings::Bucket { .. }
            | PolicySettings::CacheAware { .. }
    );
    config.routing_key_override.enabled && placed
}

/// Why a plan of `sides` sides takes no rule of the sticky override, where
/// the override places a side of it. `keyed_itself` is a side the override
/// does not place: its policy reads the routing key itself. `None` for a
/// single request's plan, and for a pair's whose two sides the override
/// places.
///
/// A binding's rule places every side of its plan, each side by its own
/// session, as the gateway's sticky map pins each role under its own key
/// (`model_gateway/src/policies/registry.rs`, `select_sticky`). The map
/// places a role at a time, so beside a side whose policy reads the key
/// itself it places the other side alone (`sticky_for`), and no document
/// holds the rule to one side yet. The check takes no sessions rule under a
/// plan with an encode side.
fn override_refused(sides: usize, keyed_itself: Option<&'static str>) -> Option<&'static str> {
    match (sides, keyed_itself) {
        (1, _) | (2, None) => None,
        (2, Some("prefill")) => Some(
            "the sticky sessions' rule places both sides of a pair, and the prefill side's policy reads the routing key itself",
        ),
        (2, Some(_)) => Some(
            "the sticky sessions' rule places both sides of a pair, and the decode side's policy reads the routing key itself",
        ),
        _ => Some(
            "the sticky sessions' rule places a single request or the two sides of a pair, and this plan has an encode side",
        ),
    }
}

/// The requests of one key a worker carries under the sticky override
/// before the key's next request is placed again: the number of the
/// gateway's own sticky map.
const OVERRIDE_INFLIGHT_CAP: u16 = 2;

/// The session rule the sticky override places a keyed request by, as the
/// gateway's own sticky map keeps a key:
///
/// - A key's first request is placed by the override's assignment: the
///   policy's own stages under `delegate`, its default.
/// - A worker carries two requests of one key. The next is placed again,
///   and the key's pin moves only to another worker under that count.
/// - Where a key's worker stops taking requests, the worker placed next
///   leads the key's pins under `delegate`. Under the map's own
///   assignments it joins behind the first, which leads again once it
///   takes requests.
/// - An idle key is forgotten after the override's idle time, at its
///   sweep, and never where either time is zero ([`pin_times`]): the
///   override keeps its keys in the manual policy's own map.
/// - A request the rule places alone is kept from the profile
///   ([`RulePlaced::Untold`]): the gateway's policy hears of no request
///   its sticky map placed
///   (`model_gateway/src/policies/registry.rs:794-797`). No other rule
///   these settings write says so.
///
/// The rule names no header: the binding's `routing_key` block lists the
/// key's sources ([`key_sources`]). It stands on a single request's binding,
/// and on a pair's where the override places both sides
/// ([`override_refused`]).
fn override_rule(config: &RoutingSettings) -> Result<SessionsSpec, RoutingSettingsError> {
    let sticky = &config.routing_key_override;
    let (ttl_s, sweep_s) = pin_times(
        ("routing_key_override.max_idle_secs", sticky.max_idle_secs),
        (
            "routing_key_override.eviction_interval_secs",
            sticky.eviction_interval_secs,
        ),
    )?;
    Ok(SessionsSpec {
        sticky_header: String::new(),
        headers: Vec::new(),
        ttl_s,
        sweep_s,
        assignment: sticky.assignment_mode,
        inflight_cap: OVERRIDE_INFLIGHT_CAP,
        issue_session_id: false,
        recovery: match sticky.assignment_mode {
            Assignment::Delegate => Recovery::Latest,
            Assignment::Random | Assignment::MinLoad | Assignment::MinGroup => Recovery::Original,
        },
        rule_placed: RulePlaced::Untold,
    })
}

/// The binding's overload block, or none where the configuration stands at
/// the block's own defaults: its switch and its shed word. The two
/// thresholds travel in no document: they are the default a gateway builds
/// its worker objects with, and a worker's registration states its own over
/// them.
fn overload(config: &RoutingSettings) -> Option<OverloadSpec> {
    // A threshold at none switches its signal off in the gateway, on its
    // worker objects. With the veto on, one signal off leaves the block on,
    // and both off is the block off.
    let waiting = config.worker_overload_waiting_requests;
    let usage = config.worker_overload_token_usage;
    let written = OverloadSpec {
        enabled: config.worker_overload_protection && (waiting.is_some() || usage.is_some()),
        shed_when_all_vetoed: config.worker_overload_shed,
        ..OverloadSpec::default()
    };
    unless(written, OverloadSpec::default())
}
