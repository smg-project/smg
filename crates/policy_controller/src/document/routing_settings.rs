//! The policy document a gateway's routing settings mean: [`RoutingSettings`],
//! the types it is made of, and why some settings have no document
//! ([`RoutingSettingsError`]).

use serde::Serialize;
use serde_json::Value;

use crate::{
    document::spec::{
        Assignment, BestWithTemperatureParams, BindingSpec, BoundedLoadParams, BucketParams,
        ConsistentHashParams, IndexKind, IndexSpec, KeySourceSpec, KvPressureParams,
        LeastLoadParams, OverloadSpec, Pairing, PickStage, PolicySpec, PrefixAffinityParams,
        ProfileSpec, RoutingKeySpec, ScoreSpec, ScoreStage, SessionsSpec, SpecError, SpillParams,
        WarmupSliceParams, MANUAL_SESSION_HEADER, MAX_KEY_SOURCES, OVERLOAD_TOKEN_USAGE,
        OVERLOAD_WAITING_REQUESTS,
    },
    engine::view::Phase,
    policies::{
        bucket,
        cache_aware::{
            Gates, DEFAULT_THRESHOLD, DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY,
            DEFAULT_WARMUP_SECS, DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
        },
        power_of_two::PowerOfTwoMetric,
    },
    stages::scorers::{
        DEFAULT_KV_PRESSURE_WEIGHT, DEFAULT_MEAN_PREFILL_TOKENS, DEFAULT_THROUGHPUT,
    },
};

/// A gateway's routing settings, and the policy document they mean.
///
/// # Why this exists
///
/// A gateway that ran routing policies of its own holds routing settings: a
/// policy by its name with its numbers, a mode, and a few settings every
/// policy shares. This type is those settings in the one form this crate
/// reads: what the gateway's flags, its configuration file or its bindings say
/// about where a request goes. The gateway maps its own configuration onto it.
/// A field carries the name the configuration gives its setting, and a
/// refusal names a setting by that name.
///
/// [`RoutingSettings::policy_document`] converts the settings into the
/// controller's [`PolicySpec`]. [`RoutingSettings::policy_document_in_force`]
/// takes a supplied document when the settings are at the gateway's defaults
/// or describe that same document. A gateway calls them at its start, and
/// applies the document in force to its store. No request path calls them.
///
/// # Where a setting lands
///
/// - A policy name lands on the controller's preset of that name. A
///   prefill-decode mode lands on the pair plan, with one profile a side.
/// - A gateway of many routers serves pairs and single workers over its
///   one document, so the document binds both: pairs first, over the
///   prefill and decode phases, and single workers behind. The sides of a
///   pair run under the mode's side policies and single workers under the
///   main policy, as under the gateway's own policies.
/// - A policy's own setting lands in the stage of the preset that reads it,
///   and only where it differs from the number that stage runs at when its
///   document names none. The controller names those numbers and this
///   module reads them, so a configuration at them is the preset itself.
/// - The overload thresholds land on every binding, the pairing mode on
///   the pair plan, the index settings under every model, and the block
///   size and the cache boundaries in the view.
/// - The manual policy's settings land in the binding's session rule.
/// - A keyed policy's key sources land on the binding. The two rings name
///   them under `routing_key`. The manual policy names them in its session
///   rule while they are headers alone, and under `routing_key` once the
///   override adds the request's own id.
/// - A number that means never lands as the largest its place holds: a
///   balance threshold past the document's width, and the sweep interval of
///   an index the configuration never sweeps. The gateway's own policy
///   trims an index at its sweep alone, so that index's size bound lands as
///   the largest too.
///
/// # What is refused
///
/// A setting that decides a placement and has no place in the document is
/// refused by its name ([`RoutingSettingsError::UnsupportedSetting`]). It is never left
/// out in silence.
///
/// # What is left out
///
/// The document of many routers plans one family where the other has no
/// document. Pairs are left out of a regular configuration under the manual
/// policy, whose session rule runs under the single plan alone, and beside
/// the encode mode. Single workers are left out of a pair configuration
/// whose main policy reads a setting no document says. A request of the
/// family left out is refused, as it was.
///
/// # What is not read
///
/// What the gateway keeps for itself under the controller is no part of the
/// document: the pools and their addresses, discovery, the probes and the
/// load poll, retries and the breaker, the prefill queue and the decode
/// rooms. The pin of a sticky key to one engine of a grouped worker is the
/// dispatch's as well: it runs behind the decision, on the worker the
/// decision names. Its key is the gateway's own, read from the routing-key
/// headers. A supplied document names its own key sources, which need not
/// be that key, so the pin is refused beside one. The copy of the cache
/// boundaries at the top of the configuration decides nothing either: each
/// policy reads its own.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingSettings {
    /// The kinds of worker the gateway routes to, with each side's own policy.
    pub mode: ModeSettings,
    /// The main policy. Single workers run under it, and so does a side of
    /// the mode that names no policy of its own.
    pub policy: PolicySettings,
    /// The gateway runs many routers side by side, so one document plans pairs
    /// and single workers.
    pub enable_igw: bool,
    /// A document supplied beside the settings, to run in place of the one
    /// they mean.
    pub policy_spec: Option<PolicySpec>,
    /// The pin of a sticky key to one engine of a grouped worker: its
    /// spelling in the configuration where the pin is on, and none where it
    /// is off.
    pub grouped_dp_sticky: Option<&'static str>,
    /// The gateway chooses a data-parallel rank itself, from the ranks'
    /// reported token loads.
    pub dp_minimum_tokens_scheduler: bool,
    /// How strictly a prefill pairs with a decode.
    pub pd_pairing_mode: Pairing,
    /// The warm-up slice of the cache-aware policy: the seconds a worker
    /// stays in it after it becomes routable.
    pub worker_warmup_secs: u64,
    /// The share of cache misses offered to the workers of the slice.
    pub worker_warmup_share: f32,
    /// The blocks a worker's index grows by before it leaves the slice.
    pub worker_warmup_blocks: usize,
    /// The share of the fleet's level under which a worker's index is thin.
    pub worker_warmup_thin_ratio: f32,
    /// One cache hit in this many is diverted to a thin worker.
    pub worker_warmup_divert_every: u64,
    /// The sticky override, and the headers a request's routing key is read
    /// from.
    pub routing_key_override: RoutingKeyOverrideSettings,
    /// The overload veto is on.
    pub worker_overload_protection: bool,
    /// The waiting requests at which a worker counts as overloaded. None
    /// switches the signal off.
    pub worker_overload_waiting_requests: Option<usize>,
    /// The KV-cache token usage at which a worker counts as overloaded. None
    /// switches the signal off.
    pub worker_overload_token_usage: Option<f64>,
    /// A request is refused where every worker it could use is overloaded.
    pub worker_overload_shed: bool,
}

impl RoutingSettings {
    /// The provider's name, where every router of the gateway is that
    /// provider's: a provider mode of a gateway that does not run many routers.
    /// Such a gateway places its requests by no routing policy.
    fn provider_alone(&self) -> Option<&'static str> {
        match self.mode {
            ModeSettings::Provider(provider) => (!self.enable_igw).then_some(provider),
            ModeSettings::Regular
            | ModeSettings::PrefillDecode { .. }
            | ModeSettings::EncodePrefillDecode { .. } => None,
        }
    }
}

/// The two settings of the sticky override this module reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingKeyOverrideSettings {
    /// A request that carries a sticky key is placed by the sticky sessions'
    /// rule, whatever the policy.
    pub enabled: bool,
    /// The header names a request's routing key is read from, in order.
    pub headers: Vec<String>,
}

/// The kinds of worker a gateway routes to. A side names its own policy or
/// runs under the main one. The pools of a mode and their addresses are the
/// gateway's own, and no part of this type.
#[derive(Clone, Debug, PartialEq)]
pub enum ModeSettings {
    /// Single workers.
    Regular,
    /// Pairs of a prefill worker and a decode worker.
    PrefillDecode {
        /// The prefill side's own policy.
        prefill_policy: Option<PolicySettings>,
        /// The decode side's own policy.
        decode_policy: Option<PolicySettings>,
    },
    /// An encode worker ahead of a pair.
    EncodePrefillDecode {
        /// The encode side's own policy.
        encode_policy: Option<PolicySettings>,
        /// The prefill side's own policy.
        prefill_policy: Option<PolicySettings>,
        /// The decode side's own policy.
        decode_policy: Option<PolicySettings>,
    },
    /// The workers of an outside provider, under the mode's name in the
    /// configuration.
    Provider(&'static str),
}

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

/// What the `power_of_two` policy compares its candidates by, as the
/// configuration says it: [`PowerOfTwoMetric`], with the price of a request
/// in the configuration's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LoadMetricSetting {
    /// The expected wait of two random workers.
    ExpectedWait,
    /// The requests in flight of two random workers.
    Requests,
    /// The requests in flight of every worker.
    LeastRequests,
    /// The resident context tokens of every worker, each request also priced
    /// at `request_tokens`.
    ResidentTokens {
        /// The price of one request, in context tokens.
        request_tokens: u64,
    },
}

/// A routing policy under its name in the configuration, with its settings.
/// A policy lands on the preset of its name ([`PolicySettings::name`]), and
/// each setting in the stage of that preset that reads it.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicySettings {
    /// `random`.
    Random,
    /// `round_robin`.
    RoundRobin,
    /// `passthrough`: one worker, and no balancing.
    Passthrough,
    /// `cache_aware`: a request goes to the worker that holds its prefix,
    /// unless the load says otherwise.
    CacheAware {
        /// The matched share of a prompt at which a request pins to the
        /// worker that holds it.
        cache_threshold: f32,
        /// The spill gate's absolute part: requests over the fleet's mean.
        balance_abs_threshold: usize,
        /// The spill gate's relative part: a multiple of the fleet's mean.
        balance_rel_threshold: f32,
        /// The price of a held prefix at the spill gate, in tokens a request.
        balance_prefix_margin_tokens: u32,
        /// A pick on a holder is credited with the prompt it still has to
        /// prefill, and not with the whole prompt.
        balance_uncached_credit: bool,
        /// The seconds between two sweeps of the index. Zero sweeps nothing.
        eviction_interval_secs: u64,
        /// The size the index is trimmed to at a sweep.
        max_tree_size: usize,
        /// The tokens of one block.
        block_size: usize,
        /// The spread of KV usage over the fleet past which affinity is
        /// given up.
        balance_token_usage_threshold: f32,
        /// The KV usage of one worker past which affinity is given up.
        overload_token_usage_threshold: f32,
        /// The decay of an overlap score by the worker's waiting prefill.
        overlap_decay: f32,
        /// The temperature of the pick. Zero takes the best candidate.
        selection_temperature: f32,
        /// The under-layer of the index.
        cache_index: IndexKind,
        /// The seconds a placement stays in the index.
        cache_ttl_secs: u64,
        /// The token positions a request's head is keyed at.
        cache_boundaries: Vec<usize>,
        /// A selection policy other than the gateway's default one. The gateway
        /// leaves its default's name out: no document says another.
        selection_policy: Option<String>,
        /// The milliseconds an optimistic booking lives. Zero books none.
        selection_accounting_ttl_ms: u64,
    },
    /// `power_of_two`.
    PowerOfTwo {
        /// What the policy compares its candidates by.
        load_metric: LoadMetricSetting,
    },
    /// `least_load`: the lowest expected wait.
    LeastLoad {
        /// The weight of KV pressure, in seconds.
        kv_pressure_weight: f64,
        /// The prefill length assumed for a request of unknown size, in
        /// tokens.
        mean_prefill_tokens: u32,
        /// The throughput of a prefill worker, and of any worker where none
        /// reports its own, in tokens a second.
        default_throughput: f64,
    },
    /// `bucket`.
    Bucket {
        /// The absolute load difference that starts balancing.
        balance_abs_threshold: usize,
        /// The relative load ratio that starts balancing.
        balance_rel_threshold: f32,
        /// The seconds between two adjustments of the bucket boundaries.
        bucket_adjust_interval_secs: usize,
    },
    /// `manual`: sticky sessions by a routing key.
    Manual {
        /// The seconds between two sweeps of the sessions.
        eviction_interval_secs: u64,
        /// The seconds a key stays pinned with no request.
        max_idle_secs: u64,
        /// How a key seen for the first time picks its worker.
        assignment_mode: Assignment,
    },
    /// `consistent_hashing`.
    ConsistentHashing,
    /// `prefix_hash`: a hash ring over a prompt's first tokens, with a bound
    /// on a worker's load.
    PrefixHash {
        /// The prefix tokens hashed.
        prefix_token_count: usize,
        /// The multiple of the mean load past which a worker is passed over.
        load_factor: f64,
        /// The load over the mean a worker must also exceed to be passed
        /// over.
        balance_abs_threshold: usize,
        /// The token positions a request's head is keyed at.
        cache_boundaries: Vec<usize>,
    },
}

impl PolicySettings {
    /// The policy's name in the configuration, which is its preset's.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::RoundRobin => "round_robin",
            Self::Passthrough => "passthrough",
            Self::CacheAware { .. } => "cache_aware",
            Self::PowerOfTwo { .. } => "power_of_two",
            Self::LeastLoad { .. } => "least_load",
            Self::Bucket { .. } => "bucket",
            Self::Manual { .. } => "manual",
            Self::ConsistentHashing => "consistent_hashing",
            Self::PrefixHash { .. } => "prefix_hash",
        }
    }
}

/// The encode side's policy where the configuration names none: encode
/// placement follows an item's content, not the request's main policy.
static ENCODE_DEFAULT: PolicySettings = PolicySettings::ConsistentHashing;

/// The encode side's profile, by its name in a document of three sides.
const ENCODE_SIDE: &str = "encode";

/// Why a set of routing settings has no policy document to run.
#[derive(Debug, PartialEq, thiserror::Error)]
pub enum RoutingSettingsError {
    /// A setting decides a placement and the document has no place for it.
    #[error("{setting} = {value}: {reason}")]
    UnsupportedSetting {
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
    /// The document's own error: a preset that does not build, or a supplied
    /// document that does not validate.
    #[error(transparent)]
    Document(#[from] SpecError),
    /// The flags change the default document and differ from the supplied one.
    #[error("the routing flags and the policy document disagree at {at}: the flags mean {flags}, the document says {document}")]
    Disagreement {
        /// Where the documents first differ, in their own field names.
        at: String,
        /// The value from the routing flags.
        flags: String,
        /// The value from the supplied document.
        document: String,
    },
}

fn unsupported(
    setting: &'static str,
    value: impl ToString,
    reason: &'static str,
) -> RoutingSettingsError {
    RoutingSettingsError::UnsupportedSetting {
        setting,
        value: value.to_string(),
        reason,
    }
}

impl RoutingSettings {
    /// The policy document these settings say, or the first of them that no
    /// document says.
    ///
    /// A pure function of the settings. The document is not validated
    /// here: the controller validates a document where it is applied, and
    /// names there a stage its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, RoutingSettingsError> {
        refuse_unsupported(self)?;
        if let Some(both) = both_families(self)? {
            match written(self, both) {
                // A setting the other family's policy reads and no document
                // says leaves that family out. The document then plans the
                // configuration's own family alone.
                Err(RoutingSettingsError::UnsupportedSetting { .. }) => {}
                said => return said,
            }
        }
        written(self, presets(self)?)
    }

    /// The document a gateway of these settings runs. Without a supplied
    /// document, it is the one the routing settings mean. A supplied
    /// document takes effect if the settings describe the same document or
    /// the document of `defaults` for this mode. `defaults` are the
    /// settings of a gateway that sets none: the gateway's own, since this crate
    /// holds no default of a gateway's configuration. A conflict names the
    /// first difference and the two values.
    ///
    /// Both launchers refuse a routing setting beside a document before
    /// this runs, so the comparison judges a configuration built by hand.
    /// The generation is no part of it: it numbers the document, and no
    /// routing setting says it. A document that names none is this
    /// gateway's first and takes number 1, since a store applies only a
    /// number above the one it holds.
    pub fn policy_document_in_force(
        &self,
        defaults: &Self,
    ) -> Result<PolicySpec, RoutingSettingsError> {
        let mut said = self.policy_document()?;
        let Some(given) = &self.policy_spec else {
            return Ok(said);
        };
        // The dispatch pins a sticky key by the gateway's own key. A supplied
        // document names its own key sources: where they are not that key,
        // the pin would split a conversation the document keeps together.
        if let Some(pin) = self.grouped_dp_sticky {
            return Err(unsupported(
                "grouped_dp_sticky",
                pin,
                "beside a supplied document the pin would read the gateway's own key, which that document's key sources need not be",
            ));
        }
        let mut given = given.clone();
        if given.generation == 0 {
            given.generation = 1;
        }
        // The default configuration of this gateway's kind: a provider mode
        // has a document under the multi-router setting alone.
        let plain = Self {
            mode: self.mode.with_defaults(),
            enable_igw: self.enable_igw,
            ..defaults.clone()
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
        Err(RoutingSettingsError::Disagreement {
            at: "document".to_owned(),
            flags: flags.to_string(),
            document: document.to_string(),
        })
    }
}

impl ModeSettings {
    /// This mode with no side policy, so the default document for this mode
    /// carries none of the configured side settings.
    fn with_defaults(&self) -> Self {
        match self {
            Self::Regular => Self::Regular,
            Self::PrefillDecode {
                prefill_policy: _,
                decode_policy: _,
            } => Self::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            Self::EncodePrefillDecode {
                encode_policy: _,
                prefill_policy: _,
                decode_policy: _,
            } => Self::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            Self::Provider(provider) => Self::Provider(provider),
        }
    }
}

/// Walk each object in the flags' written order, then through keys that
/// appear only in the supplied document. Walk lists by index.
fn first_difference(
    flags: Option<&Value>,
    document: Option<&Value>,
    at: String,
) -> Option<RoutingSettingsError> {
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
            Some(RoutingSettingsError::Disagreement {
                at,
                flags: written(flags),
                document: written(document),
            })
        }
    }
}

/// The settings that decide a placement today and that no document says.
fn refuse_unsupported(config: &RoutingSettings) -> Result<(), RoutingSettingsError> {
    if config.dp_minimum_tokens_scheduler {
        return Err(unsupported(
            "dp_minimum_tokens_scheduler",
            true,
            "the document has no rank policy: under the controller a rank is a candidate of the profile",
        ));
    }
    Ok(())
}

/// A profile of the document and the policy that decides it.
type Side<'a> = (&'static str, &'a PolicySettings);

/// A document and, binding by binding, the sides its plan runs.
type Planned<'a> = (PolicySpec, Vec<Vec<Side<'a>>>);

/// The presets the mode's policies name, as a document of one binding, with
/// its profiles: one named for the policy under the regular mode, one a side
/// under the pair and the encode modes. A provider mode of a multi-router
/// gateway reads as the regular mode: the document says the policy of its
/// self-hosted workers.
fn presets(config: &RoutingSettings) -> Result<Planned<'_>, RoutingSettingsError> {
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

/// The document of a gateway of many routers, which serves pairs and single
/// workers side by side: the pair binding first, over the prefill and decode
/// phases, and the single binding behind it for every other request. Each
/// family runs under the policies the gateway's own routing gives it. The
/// sides of a pair run under the mode's side policies, which are the main
/// policy in a regular configuration. Single workers run under the main
/// policy.
///
/// `None` where the document plans one family: a gateway of one router, the
/// encode mode, and a regular configuration whose main policy runs on no
/// side of a pair. The manual policy is that one: its picker leaves the
/// placement to the binding's session rule, and a rule runs under the single
/// plan alone.
fn both_families(config: &RoutingSettings) -> Result<Option<Planned<'_>>, RoutingSettingsError> {
    let (mode, main) = (&config.mode, &config.policy);
    let pairs = match mode {
        ModeSettings::Regular | ModeSettings::Provider(_) => {
            !matches!(main, PolicySettings::Manual { .. })
        }
        ModeSettings::PrefillDecode {
            prefill_policy: _,
            decode_policy: _,
        } => true,
        ModeSettings::EncodePrefillDecode {
            encode_policy: _,
            prefill_policy: _,
            decode_policy: _,
        } => false,
    };
    if !(config.enable_igw && pairs) {
        return Ok(None);
    }
    let (prefill, decode) = (mode.prefill_policy(main), mode.decode_policy(main));
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
    config: &'a RoutingSettings,
    (mut document, bindings): Planned<'a>,
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
            }
        }
        bound.push((sides.len(), std::mem::take(&mut shared.bound)));
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
    let overload = overload(config)?;
    for binding in &mut document.bindings {
        binding.overload.clone_from(&overload);
    }
    Ok(document)
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
    /// profiles. The headers the manual policy reads are the view's too.
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
/// side that names them.
fn one_list(
    keys: Vec<(&'static str, RoutingKeySpec)>,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let of_session = |source: &&KeySourceSpec| !source.client;
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
/// not counted.
fn within_a_block(
    key: &RoutingKeySpec,
    config: &RoutingSettings,
) -> Result<(), RoutingSettingsError> {
    if key.sources.iter().filter(|source| !source.client).count() > MAX_KEY_SOURCES {
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
    policy: &PolicySettings,
    config: &RoutingSettings,
) -> Result<Option<RoutingKeySpec>, RoutingSettingsError> {
    let overridden = config.routing_key_override.enabled;
    let hint = match policy {
        PolicySettings::Manual { .. } | PolicySettings::ConsistentHashing => true,
        PolicySettings::PrefixHash { .. } if !overridden => false,
        // Each kind is named: a new one builds once its answer is written.
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
    }
    Ok(Some(RoutingKeySpec {
        sources,
        strip_lineage: overridden,
    }))
}

/// The binding's overload block, or none where the configuration stands at
/// the block's own defaults.
fn overload(config: &RoutingSettings) -> Result<Option<OverloadSpec>, RoutingSettingsError> {
    // A threshold at none switches its signal off in the gateway. The
    // document reads an absent threshold as the controller's own number, so
    // it cannot say one signal off alone. Both off is the block off.
    let waiting = config.worker_overload_waiting_requests;
    let usage = config.worker_overload_token_usage;
    let enabled = config.worker_overload_protection && (waiting.is_some() || usage.is_some());
    if enabled {
        let one_off = "the document's overload block cannot switch one signal off alone";
        if waiting.is_none() {
            return Err(unsupported(
                "worker_overload_waiting_requests",
                "null",
                one_off,
            ));
        }
        if usage.is_none() {
            return Err(unsupported("worker_overload_token_usage", "null", one_off));
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

    use super::*;
    use crate::document::spec::{AppliedSpec, PRESETS};

    /// The settings of a gateway that sets none, as its configuration
    /// defaults them. The document they mean is the `random` preset.
    fn defaults() -> RoutingSettings {
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
                headers: vec!["x-smg-routing-key".to_string()],
            },
            worker_overload_protection: true,
            worker_overload_waiting_requests: Some(8),
            worker_overload_token_usage: Some(0.8),
            worker_overload_shed: false,
        }
    }

    /// The cache-aware policy with these five settings, and every other at
    /// the value a gateway's configuration file gets where it leaves the
    /// setting out.
    fn cache_aware(
        cache_threshold: f32,
        balance_abs_threshold: usize,
        balance_rel_threshold: f32,
        eviction_interval_secs: u64,
        max_tree_size: usize,
    ) -> PolicySettings {
        PolicySettings::CacheAware {
            cache_threshold,
            balance_abs_threshold,
            balance_rel_threshold,
            balance_prefix_margin_tokens: 0,
            balance_uncached_credit: false,
            eviction_interval_secs,
            max_tree_size,
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
    }

    /// The check a gateway passes a document through before it runs it, with
    /// the header a gateway keeps to itself denied to the view.
    fn validated(document: &PolicySpec) -> Result<AppliedSpec, SpecError> {
        AppliedSpec::validate_with(document, &["x-smg-tenant-id"])
    }

    fn config(policy: PolicySettings) -> RoutingSettings {
        RoutingSettings {
            policy,
            ..defaults()
        }
    }

    /// The policy of a preset's name with every setting at its flag's
    /// default.
    fn at_flag_defaults(name: &str) -> PolicySettings {
        match name {
            "random" => PolicySettings::Random,
            "round_robin" => PolicySettings::RoundRobin,
            "passthrough" => PolicySettings::Passthrough,
            "consistent_hashing" => PolicySettings::ConsistentHashing,
            "cache_aware" => cache_aware(0.3, 64, 1.5, 120, 67_108_864),
            "power_of_two" => PolicySettings::PowerOfTwo {
                load_metric: LoadMetricSetting::ExpectedWait,
            },
            "least_load" => PolicySettings::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
            "bucket" => PolicySettings::Bucket {
                balance_abs_threshold: 64,
                balance_rel_threshold: 1.5,
                bucket_adjust_interval_secs: 5,
            },
            "manual" => PolicySettings::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: Assignment::Random,
            },
            "prefix_hash" => PolicySettings::PrefixHash {
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

    #[test]
    fn with_no_document_the_flags_document_is_in_force() {
        let config = config(PolicySettings::RoundRobin);
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Ok(preset("round_robin"))
        );
    }

    #[test]
    fn a_given_document_wins_where_no_routing_setting_is_set() {
        for (mode, document) in [
            (ModeSettings::Regular, preset("round_robin")),
            (
                ModeSettings::PrefillDecode {
                    prefill_policy: None,
                    decode_policy: None,
                },
                PolicySpec::pair_presets("round_robin", "random").expect("paired presets"),
            ),
        ] {
            let config = RoutingSettings {
                mode,
                policy_spec: Some(document.clone()),
                ..defaults()
            };
            assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
        }
    }

    #[test]
    fn a_given_document_that_the_flags_mean_too_is_in_force() {
        let document = preset("round_robin");
        let config = RoutingSettings {
            policy_spec: Some(document.clone()),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(config.policy_document_in_force(&defaults()), Ok(document));
    }

    #[test]
    fn the_generation_numbers_the_document_and_is_no_part_of_the_comparison() {
        for (written, runs) in [(0, 1), (1, 1), (7, 7)] {
            let mut document = preset("round_robin");
            document.generation = written;
            let mut in_force = document.clone();
            in_force.generation = runs;
            for settings in [defaults(), config(PolicySettings::RoundRobin)] {
                let config = RoutingSettings {
                    policy_spec: Some(document.clone()),
                    ..settings
                };
                assert_eq!(
                    config.policy_document_in_force(&defaults()).as_ref(),
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
        let config = RoutingSettings {
            policy_spec: Some(document),
            ..config(at_flag_defaults("power_of_two"))
        };
        let refusal = config
            .policy_document_in_force(&defaults())
            .expect_err("different stages");
        assert_eq!(
            refusal.to_string(),
            r#"the routing flags and the policy document disagree at profiles.power_of_two.score[0]: the flags mean "least-load", the document says "bucket""#
        );
    }

    #[test]
    fn a_side_policy_that_says_something_else_is_refused_with_both_values() {
        let config = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(PolicySettings::RoundRobin),
                decode_policy: None,
            },
            policy_spec: Some(
                PolicySpec::pair_presets("random", "random").expect("paired presets"),
            ),
            ..defaults()
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
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
        let mut config = RoutingSettings {
            policy_spec: Some(document),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
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
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Disagreement {
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
        let config = RoutingSettings {
            policy_spec: Some(document),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            config.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::Document(SpecError::Kind(
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

    fn refusal(config: &RoutingSettings) -> (&'static str, String) {
        match config.policy_document() {
            Err(RoutingSettingsError::UnsupportedSetting { setting, value, .. }) => {
                (setting, value)
            }
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
                LoadMetricSetting::ExpectedWait,
                PowerOfTwoMetric::ExpectedWait,
            ),
            (LoadMetricSetting::Requests, PowerOfTwoMetric::Requests),
            (
                LoadMetricSetting::LeastRequests,
                PowerOfTwoMetric::LeastRequests,
            ),
            (
                LoadMetricSetting::ResidentTokens {
                    request_tokens: 512,
                },
                PowerOfTwoMetric::ResidentTokens {
                    request_tokens: 512,
                },
            ),
        ] {
            assert_eq!(
                config(PolicySettings::PowerOfTwo { load_metric }).policy_document(),
                Ok(PolicySpec::power_of_two(form).expect("a form of the preset")),
                "{load_metric:?} is the preset in that metric's form"
            );
        }
    }

    #[test]
    fn a_pair_mode_is_the_pair_of_presets_under_its_pairing_rule() {
        let pair = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(PolicySettings::RoundRobin),
                decode_policy: None,
            },
            pd_pairing_mode: Pairing::Strict,
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
        let encode_pair = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: Some(PolicySettings::Random),
            },
            ..config(PolicySettings::RoundRobin)
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
        let tuned = RoutingSettings {
            policy: PolicySettings::CacheAware {
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
                cache_index: IndexKind::Hash,
                cache_ttl_secs: 300,
                cache_boundaries: vec![512, 2048],
                selection_policy: None,
                selection_accounting_ttl_ms: 200,
            },
            worker_warmup_secs: 0,
            worker_warmup_blocks: 2048,
            ..defaults()
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
        let least_load = config(PolicySettings::LeastLoad {
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
        let bucket = config(PolicySettings::Bucket {
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
        let prefix_hash = config(PolicySettings::PrefixHash {
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
        let manual = PolicySettings::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 3600,
            assignment_mode: Assignment::MinGroup,
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
        let pinned = config(PolicySettings::Manual {
            eviction_interval_secs: 30,
            max_idle_secs: 0,
            assignment_mode: Assignment::Random,
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
        let unswept = config(PolicySettings::Manual {
            eviction_interval_secs: 0,
            max_idle_secs: 3600,
            assignment_mode: Assignment::Random,
        })
        .policy_document()
        .expect("a document");
        assert_eq!(unswept, edited(preset("manual"), for_ever));

        // A pair of manual sides: the pair of presets names no rule, and the
        // manual preset's own is the one written.
        let mut pair = PolicySpec::pair_presets("manual", "manual").expect("a pair of presets");
        pair.bindings[0].sessions = preset("manual").bindings[0].sessions.clone();
        let both = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(manual.clone())
        };
        assert_eq!(both.policy_document(), Ok(edited(pair, rule)));

        let one_side = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: Some(PolicySettings::RoundRobin),
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
        assert!(validated(&written).is_ok());

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
        assert!(validated(&document).is_ok());

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
        let mut plan = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: Some(PolicySettings::RoundRobin),
                prefill_policy: Some(PolicySettings::ConsistentHashing),
                decode_policy: Some(PolicySettings::ConsistentHashing),
            },
            ..config(PolicySettings::RoundRobin)
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
        let pair = |prefill: &str, decode: &str| RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(at_flag_defaults(prefill)),
                decode_policy: Some(at_flag_defaults(decode)),
            },
            pd_pairing_mode: Pairing::Strict,
            ..config(PolicySettings::RoundRobin)
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
        let tightened = RoutingSettings {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            worker_overload_shed: true,
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            tightened.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({
                    "waiting_requests": 32, "token_usage": 0.9, "shed_when_all_vetoed": true
                });
            })),
        );
        let off = RoutingSettings {
            worker_overload_protection: false,
            ..config(PolicySettings::RoundRobin)
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
        let round_robin = || config(PolicySettings::RoundRobin);
        let mut sticky = round_robin();
        sticky.routing_key_override.enabled = true;
        let mut selection = config(at_flag_defaults("cache_aware"));
        if let PolicySettings::CacheAware {
            selection_policy, ..
        } = &mut selection.policy
        {
            *selection_policy = Some("another".to_string());
        }
        for (refused, setting, value) in [
            (sticky, "routing_key_override.enabled", "true"),
            (
                RoutingSettings {
                    dp_minimum_tokens_scheduler: true,
                    ..round_robin()
                },
                "dp_minimum_tokens_scheduler",
                "true",
            ),
            (
                RoutingSettings {
                    mode: ModeSettings::Provider("a_provider"),
                    ..round_robin()
                },
                "mode",
                "a_provider",
            ),
            (selection, "selection_policy", "another"),
        ] {
            assert_eq!(refusal(&refused), (setting, value.to_string()));
        }
    }

    #[test]
    fn two_sides_that_name_two_values_of_one_place_are_refused() {
        let sides = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(cache_aware(0.3, 64, 1.5, 120, 67_108_864)),
                decode_policy: Some(cache_aware(0.3, 64, 1.5, 60, 67_108_864)),
            },
            ..config(PolicySettings::RoundRobin)
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
            let mut side = cache_aware(0.3, 64, 1.5, 120, 67_108_864);
            if let PolicySettings::CacheAware { block_size, .. } = &mut side {
                *block_size = blocks;
            }
            side
        };
        let blocks = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: Some(sized(16)),
                decode_policy: Some(sized(32)),
            },
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            refusal(&blocks),
            ("block_size", "16 and 32".to_string()),
            "the view holds one block size, so two sides with two have no document"
        );
    }

    #[test]
    fn a_number_past_the_documents_range_is_refused() {
        let for_ever = config(PolicySettings::Manual {
            eviction_interval_secs: 60,
            max_idle_secs: u64::from(u32::MAX),
            assignment_mode: Assignment::Random,
        });
        assert_eq!(refusal(&for_ever).0, "max_idle_secs");
        let wide = config(PolicySettings::PowerOfTwo {
            load_metric: LoadMetricSetting::ResidentTokens {
                request_tokens: u64::from(u32::MAX) + 1,
            },
        });
        assert_eq!(refusal(&wide).0, "load_metric.request_tokens");
    }

    #[test]
    fn the_pin_to_an_engine_of_a_grouped_worker_moves_no_document() {
        // The dispatch pins a sticky key behind the decision, on the worker
        // the decision names: the document is the same with the pin on.
        let pinned = RoutingSettings {
            grouped_dp_sticky: Some("hash"),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(pinned.policy_document(), Ok(preset("round_robin")));
        assert_eq!(
            pinned.policy_document_in_force(&defaults()),
            Ok(preset("round_robin"))
        );
        // The pin reads the gateway's own key, and a supplied document names
        // its own key sources: beside one the pin is refused by its name.
        let given = RoutingSettings {
            grouped_dp_sticky: Some("hash"),
            policy_spec: Some(preset("least_load")),
            ..defaults()
        };
        assert!(matches!(
            given.policy_document_in_force(&defaults()),
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "grouped_dp_sticky",
                ..
            })
        ));
    }

    #[test]
    fn a_balance_threshold_past_the_documents_width_lands_as_its_largest_number() {
        // A threshold no load reaches: the way a gateway's own policy
        // switches a spill off. The document's largest number is one too,
        // and the controller takes it.
        let never = usize::MAX;
        let bucket = config(PolicySettings::Bucket {
            balance_abs_threshold: never,
            balance_rel_threshold: 1.5,
            bucket_adjust_interval_secs: 5,
        });
        let prefix_hash = config(PolicySettings::PrefixHash {
            prefix_token_count: 256,
            load_factor: 1.25,
            balance_abs_threshold: never,
            cache_boundaries: Vec::new(),
        });
        let cache_aware = config(cache_aware(0.3, never, 1.5, 120, 67_108_864));
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
            assert_eq!(validated(&document).err(), None, "{at}");
        }
    }

    #[test]
    fn an_index_the_configuration_never_sweeps_is_never_trimmed() {
        // The gateway's own policy trims its index at its sweep alone, and
        // sweeps nothing at an interval of zero: the configured size bounds
        // nothing. The document says so with its longest cadence and its
        // largest size, which no write of the controller's record meets.
        let swept_every = |secs| cache_aware(0.3, 64, 1.5, secs, 67_108_864);
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
        assert_eq!(validated(&document).err(), None);
        // A swept index keeps the size it is trimmed to.
        let swept = config(cache_aware(0.3, 64, 1.5, 60, 1024))
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
        let sides = RoutingSettings {
            mode: pair_mode(Some(swept_every(0)), Some(swept_every(60))),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(
            refusal(&sides).1,
            r#"["tree",180,67108864,0] and ["tree",180,67108864,60]"#
        );
    }

    /// `config` as a gateway of many routers runs it.
    fn many_routers(config: RoutingSettings) -> RoutingSettings {
        RoutingSettings {
            enable_igw: true,
            ..config
        }
    }

    fn pair_mode(prefill: Option<PolicySettings>, decode: Option<PolicySettings>) -> ModeSettings {
        ModeSettings::PrefillDecode {
            prefill_policy: prefill,
            decode_policy: decode,
        }
    }

    #[test]
    fn the_default_document_of_many_routers_binds_pairs_ahead_of_single_workers() {
        let document = many_routers(defaults())
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
            let pair = RoutingSettings {
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
            assert_eq!(validated(&both).err(), None, "{name}");
        }
    }

    #[test]
    fn a_setting_of_the_binding_lands_on_both_bindings_of_many_routers() {
        let mut keyed = RoutingSettings {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            worker_overload_shed: true,
            pd_pairing_mode: Pairing::Strict,
            ..config(PolicySettings::ConsistentHashing)
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
        for mode in [ModeSettings::Regular, pair_mode(None, None)] {
            // No routing setting is set: the given document is in force.
            let unset = many_routers(RoutingSettings {
                mode: mode.clone(),
                policy_spec: Some(given.clone()),
                ..defaults()
            });
            assert_eq!(
                unset.policy_document_in_force(&defaults()),
                Ok(given.clone())
            );

            // A setting that says another document is refused, and one that
            // says the given document is not.
            let set = many_routers(RoutingSettings {
                mode,
                ..config(PolicySettings::RoundRobin)
            });
            let said = set.policy_document().expect("the flags' document");
            let other = RoutingSettings {
                policy_spec: Some(given.clone()),
                ..set.clone()
            };
            assert!(matches!(
                other.policy_document_in_force(&defaults()),
                Err(RoutingSettingsError::Disagreement { .. })
            ));
            let same = RoutingSettings {
                policy_spec: Some(said.clone()),
                ..set
            };
            assert_eq!(same.policy_document_in_force(&defaults()), Ok(said));
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
        let encode = RoutingSettings {
            mode: ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(PolicySettings::RoundRobin)
        };
        assert!(encode.policy_document().is_ok());
        assert_eq!(
            many_routers(encode.clone()).policy_document(),
            encode.policy_document()
        );

        // Under the override the main policy's keyed requests have no
        // document yet, while the hash ring on both sides of the pair has
        // one. The pair configuration's document then plans pairs alone.
        let mut keyed = RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::ConsistentHashing),
                Some(PolicySettings::ConsistentHashing),
            ),
            ..config(PolicySettings::RoundRobin)
        };
        keyed.routing_key_override.enabled = true;
        keyed.routing_key_override.headers = vec![MANUAL_SESSION_HEADER.to_string()];
        let pairs_alone = keyed.policy_document();
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(keyed.clone()).policy_document(), pairs_alone);
        // The same main policy alone is refused by the setting's name.
        keyed.mode = ModeSettings::Regular;
        assert_eq!(
            refusal(&many_routers(keyed)),
            ("routing_key_override.enabled", "true".to_string())
        );
    }

    /// A pair configuration with round robin on both sides, under `main`.
    fn round_robin_pairs_under(main: PolicySettings) -> RoutingSettings {
        RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::RoundRobin),
                Some(PolicySettings::RoundRobin),
            ),
            ..config(main)
        }
    }

    /// The cache-aware policy at its flags' defaults, over blocks of `blocks`.
    fn cache_aware_over(blocks: usize) -> PolicySettings {
        let mut policy = at_flag_defaults("cache_aware");
        if let PolicySettings::CacheAware { block_size, .. } = &mut policy {
            *block_size = blocks;
        }
        policy
    }

    #[test]
    fn many_routers_write_the_view_and_index_of_a_main_policy_no_side_runs() {
        // Round robin places both sides of a pair and names no view and no
        // index. A tuned cache-aware policy places single workers: what it
        // shares with every side lands in the document all the same.
        let mut main = cache_aware(0.3, 64, 1.5, 60, 16_777_216);
        if let PolicySettings::CacheAware { block_size, .. } = &mut main {
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
        assert_eq!(validated(&both).err(), None);
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
        assert_eq!(validated(&both).err(), None);
    }

    #[test]
    fn many_routers_plan_pairs_alone_where_the_main_policy_names_another_shared_value() {
        // The view holds one block size. The sides agree on theirs and the
        // main policy names another, so single workers have no document and
        // the pair configuration plans its own family alone.
        let clash = RoutingSettings {
            mode: pair_mode(Some(cache_aware_over(16)), Some(cache_aware_over(16))),
            ..config(cache_aware_over(32))
        };
        let pairs_alone = clash.policy_document();
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(clash).policy_document(), pairs_alone);
    }
}
