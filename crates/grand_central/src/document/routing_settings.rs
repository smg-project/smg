//! The policy document a gateway's routing settings mean: [`RoutingSettings`],
//! the types it is made of, and why some settings have no document
//! ([`RoutingSettingsError`]).

#![expect(
    clippy::exhaustive_structs,
    reason = "a caller still builds or matches these types whole: each mark comes with its call"
)]

mod translation;

use std::collections::BTreeMap;

use serde_json::Value;

use self::translation::{both_families, presets, three_sided, written, written_from, written_with};
// The metric is named only by a doc link of this module and by its cases.
#[cfg(any(test, doc))]
use crate::policies::power_of_two::PowerOfTwoMetric;
use crate::{
    document::spec::{
        Assignment, IndexKind, Pairing, PolicySpec, SpecError, MANUAL_SESSION_HEADER,
        OVERLOAD_WAITING_REQUESTS,
    },
    policies::cache_aware::{
        DEFAULT_WARMUP_BLOCKS, DEFAULT_WARMUP_DIVERT_EVERY, DEFAULT_WARMUP_SECS,
        DEFAULT_WARMUP_SHARE, DEFAULT_WARMUP_THIN_RATIO,
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
/// - A gateway in the prefill-decode mode places one single request beside
///   its pairs over HTTP, a token count that takes one prefill worker, and
///   a gateway of many routers serves pairs and single workers over its one
///   document. Either document binds both: pairs first, over the prefill
///   and decode phases, and single requests behind. The sides of a pair run
///   under the mode's side policies and single requests under the main
///   policy, as under the gateway's own policies. Over gRPC or ZMQ the mode
///   counts tokens with no worker, so a gateway of one router plans the
///   pair alone there ([`ConnectionModeSetting`]).
/// - A gateway of many routers mounts its three-sided router in every
///   mode, and a model with an encode pool goes to it. Its document's
///   first binding therefore plans three sides, over the encode, prefill
///   and decode phases: the pair's two as above, and an encode side under
///   the mode's encode policy, or under `consistent_hashing` where the mode
///   names none.
/// - A policy's own setting lands in the stage of the preset that reads it,
///   and only where it differs from the number that stage runs at when its
///   document names none. The controller names those numbers and this
///   module reads them, so a configuration at them is the preset itself.
/// - The overload veto's switch and its shed word land on every binding,
///   the pairing mode on the pair plan, the index settings under every
///   model, and the block size and the cache boundaries in the view.
/// - The manual policy's settings land in the binding's session rule. The
///   rule places a single request, or the two sides of a pair, with an
///   encode side ahead of them or with none. Both sides of the pair are
///   manual or neither is, and a manual encode side is refused by its
///   setting's name: no rule places an item.
/// - A keyed policy's key sources land on the binding. The two rings name
///   them under `routing_key`. The manual policy names them in its session
///   rule while they are headers alone, and under `routing_key` once the
///   override adds the request's own id.
/// - The sticky override lands as a session rule on the binding of every
///   policy but `manual` and `consistent_hashing`, which place by the key
///   themselves (`override_places` says why of each):
///   the override's own assignment and times, its two requests of a key a
///   worker, and the configured headers then the request's own id under
///   `routing_key`. A rule with a cap runs under the single plan alone,
///   so it has a document in a gateway of one router that serves single
///   requests.
/// - A number that means never lands as the largest its place holds: a
///   balance threshold past the document's width, and the sweep interval of
///   an index the configuration never sweeps. The gateway's own policy
///   trims an index at its sweep alone, so that index's size bound lands as
///   the largest too.
/// - A worker's `policy` label is no setting, and a host hands it beside
///   them ([`RoutingSettings::policy_document_with_labels`]). It lands as a
///   binding of its model's single requests, ahead of the binding of every
///   model, under the policy it names.
///
/// # What is refused
///
/// A setting that decides a placement and has no place in the document is
/// refused by its name ([`RoutingSettingsError::UnsupportedSetting`]). It is never left
/// out in silence, but for what the gateway keeps for itself (below).
///
/// # What is left out
///
/// A document of two families plans one where the other has no document.
/// Pairs are left out of a regular configuration of many routers under the
/// manual policy, whose pairs this module does not plan beside its single
/// requests yet, and beside the encode mode. Single requests are left out
/// of a pair gateway of one router that reaches its workers over gRPC or
/// ZMQ: it places none.
/// They are left out of a pair configuration whose main policy reads a
/// setting no document says: a gateway of one router then refuses its token
/// count, which its own policies place. That gateway leaves them out as well
/// where one family's policy names a block size and the other places by a
/// prefix ring, which counts its key in another size. The encode mode of
/// many routers leaves them out where the controller's check refuses the
/// document of both families, whatever it refuses: a main prefix ring the
/// view carries no block for and a main policy setting the build does not
/// run are two such. The mode's own document is checked where it is applied.
/// A request of the family left out is refused.
///
/// The encode side is left out of a gateway of many routers, outside the
/// encode mode, where the plan of three sides has no document. The
/// settings then say what they said before that plan, and an encode
/// request meets no binding with an encode side. The sticky override's
/// rule places single requests alone, so under a policy it places such a
/// gateway is refused by the setting's name, as it was. A document of
/// three sides the controller's check refuses is left out too.
///
/// # What is not read
///
/// What the gateway keeps for itself under the controller is no part of the
/// document: the pools and their addresses, discovery, the probes and the
/// load poll, retries and the breaker, the prefill queue and the decode
/// rooms. The two overload thresholds are the gateway's too: the default it
/// builds its worker objects with, under each worker's own registration. No
/// document carries a threshold, so one off its default means the same
/// document. Both at none leave no default to veto by, and the settings' own
/// document switches the block off. Beside a supplied document the two are
/// no difference, whatever they are: its host says what a block that is on
/// means over no threshold. The pin of a sticky key to one engine of a
/// grouped worker is the dispatch's as well: it runs behind the decision, on
/// the worker the decision names. Its key is the gateway's own, read from
/// the routing-key headers. A supplied document names its own key sources,
/// which need not be that key, so the pin is refused beside one. The copy
/// of the cache boundaries at the top of the configuration decides nothing
/// either: each policy reads its own.
#[derive(Clone, Debug, PartialEq)]
pub struct RoutingSettings {
    /// The kinds of worker the gateway routes to, with each side's own policy.
    pub mode: ModeSettings,
    /// The main policy. Single workers run under it, and so does a side of
    /// the mode that names no policy of its own. A worker's `policy` label
    /// names another one for its model's single requests
    /// ([`RoutingSettings::policy_document_with_labels`]).
    pub policy: PolicySettings,
    /// The gateway runs many routers side by side, so one document plans pairs
    /// and single workers.
    pub enable_igw: bool,
    /// How the gateway reaches its workers. A pair mode of one router places
    /// a single request, its token count, over HTTP alone.
    pub connection_mode: ConnectionModeSetting,
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
    /// switches the signal off. The gateway's own: no document carries it.
    pub worker_overload_waiting_requests: Option<usize>,
    /// The KV-cache token usage at which a worker counts as overloaded. None
    /// switches the signal off. The gateway's own: no document carries it.
    pub worker_overload_token_usage: Option<f64>,
    /// A request is refused where every worker it could use is overloaded.
    pub worker_overload_shed: bool,
}

impl Default for RoutingSettings {
    /// The settings of a gateway that sets none: each setting at the value it
    /// has where a configuration leaves it out. The document they mean is the
    /// `random` preset.
    ///
    /// The warm-up numbers are the ones the controller's stages run at when
    /// a document names none, so settings at them write none into a
    /// document. The overload thresholds are the crate's two defaults, and
    /// no document carries one at any value.
    fn default() -> Self {
        Self {
            mode: ModeSettings::Regular,
            policy: PolicySettings::Random,
            enable_igw: false,
            connection_mode: ConnectionModeSetting::Http,
            policy_spec: None,
            grouped_dp_sticky: None,
            dp_minimum_tokens_scheduler: false,
            pd_pairing_mode: Pairing::Lenient,
            worker_warmup_secs: DEFAULT_WARMUP_SECS,
            worker_warmup_share: DEFAULT_WARMUP_SHARE,
            worker_warmup_blocks: DEFAULT_WARMUP_BLOCKS as usize,
            worker_warmup_thin_ratio: DEFAULT_WARMUP_THIN_RATIO,
            worker_warmup_divert_every: DEFAULT_WARMUP_DIVERT_EVERY,
            routing_key_override: RoutingKeyOverrideSettings::default(),
            worker_overload_protection: true,
            worker_overload_waiting_requests: Some(OVERLOAD_WAITING_REQUESTS as usize),
            // A configuration writes this usage as 0.8 in 64 bits. The
            // controller's number is 32 bits wide: widened, it is another
            // value.
            worker_overload_token_usage: Some(0.8),
            worker_overload_shed: false,
        }
    }
}

impl RoutingSettings {
    /// The provider's name, where every router of the gateway is that
    /// provider's: a provider mode of a gateway that does not run many routers.
    /// It places by no routing policy: a host asks here ahead of the document.
    pub fn provider_alone(&self) -> Option<&'static str> {
        match self.mode {
            ModeSettings::Provider(provider) => (!self.enable_igw).then_some(provider),
            ModeSettings::Regular
            | ModeSettings::PrefillDecode { .. }
            | ModeSettings::EncodePrefillDecode { .. } => None,
        }
    }
}

/// How a gateway reaches its workers, as the configuration names it. One
/// thing follows from it here: whether a gateway of one router in the pair
/// mode places a single request, which its document then binds
/// ([`RoutingSettings`], where a setting lands).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConnectionModeSetting {
    /// Over HTTP. A pair mode counts tokens on one prefill worker, a single
    /// request its own routing places by the main policy.
    Http,
    /// Over gRPC. A pair mode counts tokens with no worker.
    Grpc,
    /// Over ZMQ, by the router that serves gRPC: no worker counts tokens.
    Zmq,
}

/// The settings of the sticky override this module reads.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RoutingKeyOverrideSettings {
    /// A request that carries a sticky key is placed by the sticky sessions'
    /// rule, whatever the policy.
    pub enabled: bool,
    /// The header names a request's routing key is read from, in order.
    pub headers: Vec<String>,
    /// The seconds between two sweeps of the override's sessions. Zero here
    /// or in `max_idle_secs` keeps every key the session table holds pinned
    /// for ever: a table full of such keys records no new key.
    pub eviction_interval_secs: u64,
    /// The seconds with no request after which a key of the override leaves
    /// at the next sweep. Zero here or in `eviction_interval_secs` keeps
    /// every key the session table holds pinned for ever: a table full of
    /// such keys records no new key.
    pub max_idle_secs: u64,
    /// How a key the override sees for the first time picks its worker.
    pub assignment_mode: Assignment,
}

impl Default for RoutingKeyOverrideSettings {
    /// The override of a gateway that sets none of it: off, with the header,
    /// the two times and the assignment a configuration that leaves them out
    /// gets.
    fn default() -> Self {
        Self {
            enabled: false,
            headers: vec![MANUAL_SESSION_HEADER.to_owned()],
            eviction_interval_secs: 60,
            max_idle_secs: 4 * 3600,
            assignment_mode: Assignment::Delegate,
        }
    }
}

/// The kinds of worker a gateway routes to. A side names its own policy or
/// runs under the main one. The pools of a mode and their addresses are the
/// gateway's own, and no part of this type.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
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

/// What the `power_of_two` policy compares its candidates by, as the
/// configuration says it: [`PowerOfTwoMetric`], with the price of a request
/// in the configuration's width.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
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
#[non_exhaustive]
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

    /// The policy a worker's `policy` label names, by the name of its
    /// preset: that policy with the settings a gateway's own registry
    /// builds it with from the name alone (`PolicyFactory::create_by_name`,
    /// `model_gateway/src/policies/factory.rs:118-134`). `None` for a name
    /// with no preset.
    ///
    /// A label carries a name and no setting, so the registry builds each
    /// policy at its type's own defaults, and a labelled model runs those.
    /// A label that names the main policy is the one exception: the
    /// registry builds that one from the settings, and the builder never
    /// asks this table for it. For eight names the defaults are the values
    /// of the name's preset. For two they are not, by seven numbers:
    ///
    /// - `cache_aware` (`CacheAwareConfig::default`,
    ///   `model_gateway/src/policies/mod.rs:359-385`): a cache threshold of
    ///   0.5, a balance threshold of 32 requests and of 1.1 times the mean,
    ///   a sweep every 30 seconds and an index trimmed to 10,000. Its preset
    ///   runs 0.3, 64, 1.5, 120 seconds and 67,108,864.
    /// - `bucket` (`BucketConfig::default`,
    ///   `model_gateway/src/policies/mod.rs:394-402`): a balance rule that
    ///   starts at a load difference of 32 booked tokens, or characters,
    ///   and a load ratio of 1.0001. Its preset runs 64 and 1.5.
    pub fn of_label(name: &str) -> Option<Self> {
        Some(match name {
            "random" => Self::Random,
            "round_robin" => Self::RoundRobin,
            "passthrough" => Self::Passthrough,
            "consistent_hashing" => Self::ConsistentHashing,
            "power_of_two" => Self::PowerOfTwo {
                load_metric: LoadMetricSetting::ExpectedWait,
            },
            "least_load" => Self::LeastLoad {
                kv_pressure_weight: 0.15,
                mean_prefill_tokens: 1024,
                default_throughput: 2000.0,
            },
            "bucket" => Self::Bucket {
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.0001,
                bucket_adjust_interval_secs: 5,
            },
            "manual" => Self::Manual {
                eviction_interval_secs: 60,
                max_idle_secs: 14_400,
                assignment_mode: Assignment::Random,
            },
            "prefix_hash" => Self::PrefixHash {
                prefix_token_count: 256,
                load_factor: 1.25,
                balance_abs_threshold: 10,
                cache_boundaries: Vec::new(),
            },
            "cache_aware" => Self::CacheAware {
                cache_threshold: 0.5,
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.1,
                balance_prefix_margin_tokens: 0,
                balance_uncached_credit: false,
                eviction_interval_secs: 30,
                max_tree_size: 10_000,
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
            },
            _ => return None,
        })
    }
}

/// Why a set of routing settings has no policy document to run.
#[derive(Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
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
    /// A worker's `policy` label that the document of these settings has no
    /// binding for. The call that refuses it answers no document. The
    /// refusal names the label alone, so a host leaves that one out and
    /// calls again ([`RoutingSettings::policy_document_with_labels`]). It
    /// is never a parting of the presets and this module, which is
    /// answered as itself under a label too.
    #[error("the policy label of model {model:?} names {policy:?}: {reason}")]
    Label {
        /// The model the label's worker serves.
        model: String,
        /// The policy the label names.
        policy: String,
        /// Why the document has no binding for it.
        reason: String,
    },
}

/// The document a plan was written into, where that plan is taken: `None`
/// where the next plan stands in its place. A setting that the plan's
/// other family reads and that no document says leaves that plan. So does
/// a document the controller's check refuses, for a plan that is `checked`:
/// one that binds more than the gateway's document bound before it, so the
/// binding costs the gateway no document that was taken.
fn taken(
    said: Result<PolicySpec, RoutingSettingsError>,
    checked: bool,
) -> Option<Result<PolicySpec, RoutingSettingsError>> {
    match said {
        Err(RoutingSettingsError::UnsupportedSetting { .. }) => None,
        Ok(document) if checked && document.validate().is_err() => None,
        said => Some(said),
    }
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
    /// A gateway of many routers mounts its three-sided router in every
    /// mode, so its document plans the three sides first: one `epd`
    /// binding ahead of the main policy's own. Where a setting has no
    /// document under that plan, or the controller's check refuses the
    /// plan's document, the settings say what they said before it: no
    /// gateway loses a document it has, and a refusal keeps its words. The
    /// sticky override, under a policy it places, is refused by the
    /// setting's name, as it was. Where the check refuses the document of
    /// three sides the pair's plan stands, and an encode request meets no
    /// binding with an encode side.
    ///
    /// The manual policy on both sides of a pair runs under the plan of
    /// three sides, and under the encode mode's own plan. Its rule places
    /// the pair's two sides, each by its own session, and the encode side
    /// places an item by its own policy.
    ///
    /// A pure function of the settings. The document is not validated
    /// here, a plan with an encode side in a gateway of many routers
    /// apart: the controller validates a document where it is applied, and
    /// names there a stage its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, RoutingSettingsError> {
        refuse_unsupported(self)?;
        if let Some(three) = three_sided(&self.mode, &self.policy, self.enable_igw)? {
            if let Some(said) = taken(written(self, three), true) {
                return said;
            }
        }
        if let Some(both) = both_families(self)? {
            let encode_mode = matches!(self.mode, ModeSettings::EncodePrefillDecode { .. });
            if let Some(said) = taken(written(self, both), encode_mode) {
                return said;
            }
        }
        written(self, presets(self)?)
    }

    /// [`Self::policy_document`] where workers' `policy` labels name the
    /// policy of their model, as a gateway's own registry reads them: a
    /// labelled model's single requests are placed by the label's policy.
    /// `labels` holds each such model's id with the name of the policy's
    /// preset. Its host resolves a label's spelling to that name, and says
    /// which registration's label stands for a model and until when.
    ///
    /// A label's binding is one, of its model alone, right ahead of the
    /// binding of every model's single requests and behind a pair's or an
    /// encode's, in the order of the models' ids. The binding is the one
    /// these settings would say for single requests under that policy, at
    /// the settings a label builds it with ([`PolicySettings::of_label`]):
    /// the policy's own key sources and session rule, the sticky override's
    /// rule where the override places the policy, the warm-up settings in
    /// its profile, and the settings' overload block. The settings' own
    /// profiles are named for a side or for the main policy, and a label's
    /// for its policy. So one profile serves every model that names one
    /// policy, and no name holds two.
    ///
    /// Models that share a profile run the same stages, so a stage that
    /// keeps a history of its own keeps one for all of them. Two stock
    /// stages keep one: round robin draws the start of a list it sees first
    /// from a count of its own, and the cache-aware warm-up counts its
    /// misses and its hits and marks a shallow hit. A model whose label
    /// names the main policy shares the main profile's the same way. A
    /// gateway's own registry builds a policy for each labelled model, so
    /// there each model keeps its own. That difference stays while one
    /// profile serves every model of a policy.
    ///
    /// Some labels add nothing. With no label this is
    /// [`Self::policy_document`]. A label that names the main policy leaves
    /// its model under the main policy's profile, with the settings'
    /// options, whatever the model's id. And where the settings' document
    /// binds no single request, no label is bound: a model's own binding
    /// would place the requests that document refuses. A label's id and its
    /// name are checked there as everywhere. What its binding would cost
    /// the document is checked only where the binding is written.
    ///
    /// The document's view is one for every binding. The headers a label's
    /// side lists there join the list behind the ones that stand in it, in
    /// the labels' order, each once. They are the key headers of a `manual`
    /// label, and the sticky override's where the override places the
    /// label's policy.
    ///
    /// The builder writes one block size and one list of cache boundaries
    /// for the document. So a label whose policy names another value than a
    /// side of the settings is refused, as two sides of the settings are. A
    /// label's prefix ring counts its key in that block size under every
    /// gateway. In a gateway of many routers the rings of the settings' own
    /// sides stand as they stood, and a label's is new.
    ///
    /// A label's index is not shared. It stands under the label's own model
    /// id, where it differs from what the document answers for that model
    /// without it: the settings' one entry, or the defaults.
    ///
    /// A label the document has no binding for is refused by its model and
    /// its policy ([`RoutingSettingsError::Label`]): a model's id that is
    /// empty or holds a `*`, a name with no preset, a policy that cannot
    /// stand beside the settings' own, and headers that would take the
    /// view's list past its bound. The call then answers that refusal and
    /// no document: a host leaves the label out and calls again. The
    /// settings' own document is decided first, with no label, so a label
    /// never changes which families it binds. Two labels never refuse one
    /// another: each stands or falls beside the settings alone, so every
    /// subset of labels that build builds too. A preset this module has
    /// parted from is no label's refusal: it is answered as itself
    /// ([`RoutingSettingsError::NoStage`], [`RoutingSettingsError::NoPart`],
    /// [`RoutingSettingsError::Document`]), under a label too.
    ///
    /// A pure function of the settings and the labels. Like
    /// [`Self::policy_document`] it answers the document the settings mean
    /// and reads no supplied document: where one is in force
    /// ([`Self::policy_document_in_force`]), that document decides, and a
    /// host hands this function no label. It does not run the document's
    /// own check on what the labels add: that check judges the whole
    /// document and names no label, so a host that adds one label at a time
    /// knows whose document it refused.
    pub fn policy_document_with_labels(
        &self,
        labels: &BTreeMap<String, String>,
    ) -> Result<PolicySpec, RoutingSettingsError> {
        let document = self.policy_document()?;
        let labels = labelled(self, labels)?;
        if labels.is_empty() {
            return Ok(document);
        }
        // The plan that document was written from ([`written_from`]),
        // written again with the labels' sides.
        let plan = match three_sided(&self.mode, &self.policy, self.enable_igw)? {
            Some(three) if written_from(&three, &document) => three,
            _ => match both_families(self)? {
                Some(both) if written_from(&both, &document) => both,
                _ => presets(self)?,
            },
        };
        written_with(self, plan, &labels)
    }

    /// The document a gateway of these settings runs. Without a supplied
    /// document, it is the one the routing settings mean. A supplied
    /// document takes effect if the settings describe the same document or
    /// the document of `defaults` for this mode. `defaults` are the
    /// settings of a gateway that sets none: [`RoutingSettings::default`],
    /// or the gateway's own where its configuration defaults a setting to
    /// another value. A conflict names the first difference and the two
    /// values.
    ///
    /// The two overload thresholds are no part of the comparison with the
    /// document of `defaults`: no document carries one, so beside settings
    /// that state nothing else they stand beside a supplied document
    /// whatever they are, both at none too. Settings that state a policy of
    /// their own are held to their whole document, whose block is off for
    /// both at none. What a supplied document's switch means over no
    /// threshold is its host's to say.
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
        // has a document under the multi-router setting alone, and a pair
        // mode binds a single request by how the gateway reaches its workers.
        // The two overload thresholds are the gateway's own beside a
        // document, so the default configuration states them as they are.
        let plain = Self {
            mode: self.mode.with_defaults(),
            enable_igw: self.enable_igw,
            connection_mode: self.connection_mode,
            worker_overload_waiting_requests: self.worker_overload_waiting_requests,
            worker_overload_token_usage: self.worker_overload_token_usage,
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

/// A model whose worker's `policy` label names a policy, with that policy
/// as a label builds it ([`PolicySettings::of_label`]).
type Labelled<'a> = (&'a str, PolicySettings);

/// The labels a document binds, in the order of their models' ids. A label
/// that names the main policy is none of them, whatever its model: that
/// model runs the main policy's profile already, with the settings'
/// options. Of the others, a model's id no binding can name alone is
/// refused, and so is a name with no preset.
fn labelled<'a>(
    config: &RoutingSettings,
    labels: &'a BTreeMap<String, String>,
) -> Result<Vec<Labelled<'a>>, RoutingSettingsError> {
    let mut bound = Vec::with_capacity(labels.len());
    for (model, name) in labels {
        let refused = |reason: String| RoutingSettingsError::Label {
            model: model.clone(),
            policy: name.clone(),
            reason,
        };
        if name == config.policy.name() {
            continue;
        }
        if model.is_empty() {
            return Err(refused(
                "a binding names its model by the model's id, and this id is empty".to_owned(),
            ));
        }
        if model.contains('*') {
            return Err(refused(
                "a binding reads a `*` at the start or the end of a model's name as a pattern and the document refuses one anywhere else, so no binding names this model alone"
                    .to_owned(),
            ));
        }
        let policy = PolicySettings::of_label(name)
            .ok_or_else(|| refused(SpecError::UnknownPreset(name.clone()).to_string()))?;
        bound.push((model.as_str(), policy));
    }
    Ok(bound)
}

#[cfg(test)]
#[path = "../../tests/unit/document/routing_settings/tests.rs"]
mod tests;
