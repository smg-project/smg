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

#[cfg(test)]
use self::translation::label_refused;
use self::translation::{both_families, presets, written, written_with};
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
#[cfg(test)]
use crate::{
    document::spec::{BindingSpec, IndexSpec, OverloadSpec, Recovery},
    engine::view::Phase,
    policies::cache_aware::Gates,
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
/// - A policy's own setting lands in the stage of the preset that reads it,
///   and only where it differs from the number that stage runs at when its
///   document names none. The controller names those numbers and this
///   module reads them, so a configuration at them is the preset itself.
/// - The overload veto's switch and its shed word land on every binding,
///   the pairing mode on the pair plan, the index settings under every
///   model, and the block size and the cache boundaries in the view.
/// - The manual policy's settings land in the binding's session rule.
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
    /// here, the encode mode of many routers apart: the controller
    /// validates a document where it is applied, and names there a stage
    /// its build does not run yet.
    pub fn policy_document(&self) -> Result<PolicySpec, RoutingSettingsError> {
        refuse_unsupported(self)?;
        if let Some(both) = both_families(self)? {
            let encode_mode = matches!(self.mode, ModeSettings::EncodePrefillDecode { .. });
            match written(self, both) {
                // A setting the other family's policy reads and no document
                // says leaves that family out. The document then plans the
                // configuration's own family alone.
                Err(RoutingSettingsError::UnsupportedSetting { .. }) => {}
                // The encode mode of many routers ran its own family alone
                // before it bound a single request. A document of both that
                // the controller's check refuses leaves the single family
                // out as well, so that binding costs the mode no document
                // that was taken.
                Ok(both) if encode_mode && both.validate().is_err() => {}
                said => return said,
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
        // The plan that document was written from, written again with the
        // labels' sides. The plan of two families binds two bindings and
        // the mode's own plan one, so the count says which it was.
        let plan = match both_families(self)? {
            Some(both) if both.1.len() == document.bindings.len() => both,
            _ => presets(self)?,
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
mod tests {
    use serde_json::json;

    use super::*;
    use crate::{
        document::spec::{AppliedSpec, PRESETS},
        engine::framework::{InTheWay, Place, Plan},
    };

    /// The settings of a gateway that sets none, as its configuration
    /// defaults them. The document they mean is the `random` preset.
    fn defaults() -> RoutingSettings {
        RoutingSettings::default()
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

    /// The settings a configuration leaves out are those of a gateway that
    /// sets none: the `random` preset, with nothing written over it.
    #[test]
    fn the_settings_of_a_gateway_that_sets_none_mean_the_random_preset() {
        assert_eq!(
            RoutingSettings::default().policy_document(),
            Ok(preset("random"))
        );
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

    /// A threshold at none switches its signal off in the gateway, on its
    /// worker objects, and no document carries a threshold. With the veto
    /// on, one signal off leaves the block switched on, whatever the other
    /// is: the settings mean the preset, with nothing written over it. Both
    /// off is the block switched off, and the shed word stays where it is
    /// written.
    #[test]
    fn one_overload_signal_off_leaves_the_block_on_and_both_off_switch_it_off() {
        let thresholds = |waiting: Option<usize>, usage: Option<f64>| {
            let mut config = config(at_flag_defaults("round_robin"));
            config.worker_overload_waiting_requests = waiting;
            config.worker_overload_token_usage = usage;
            config
        };
        for (waiting, usage) in [
            (None, Some(0.8)),
            (None, Some(0.5)),
            (Some(8), None),
            (Some(32), None),
        ] {
            assert_eq!(
                thresholds(waiting, usage).policy_document(),
                Ok(preset("round_robin")),
                "waiting {waiting:?}, usage {usage:?}"
            );
        }
        let switched_off = edited(preset("round_robin"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        });
        assert_eq!(
            thresholds(None, None).policy_document(),
            Ok(switched_off.clone())
        );
        // With the protection off the thresholds decide nothing: the block
        // is off whatever they are.
        let mut unprotected = thresholds(None, Some(0.8));
        unprotected.worker_overload_protection = false;
        assert_eq!(unprotected.policy_document(), Ok(switched_off));
        // The shed word does not switch the block on.
        let mut shedding = thresholds(None, None);
        shedding.worker_overload_shed = true;
        assert_eq!(
            shedding.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] =
                    json!({ "enabled": false, "shed_when_all_vetoed": true });
            }))
        );
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
            Ok(with_single(expected, preset("least_load"))),
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
        // manual preset's own is the one written. The single binding behind
        // the pair is the regular document's, under the same rule.
        let mut pair = PolicySpec::pair_presets("manual", "manual").expect("a pair of presets");
        pair.bindings[0].sessions = preset("manual").bindings[0].sessions.clone();
        let both = RoutingSettings {
            mode: ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            },
            ..config(manual.clone())
        };
        assert_eq!(
            both.policy_document(),
            Ok(with_single(
                edited(pair, rule),
                edited(preset("manual"), rule)
            ))
        );

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
                    { "header": "cookie", "client": true },
                    { "header": "x-smg-target-worker", "target": true }
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
                { "header": "cookie", "client": true },
                { "header": "x-smg-target-worker", "target": true }
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
                        { "header": "cookie", "client": true },
                        { "header": "x-smg-target-worker", "target": true }
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
        assert_eq!(sources, Ok(Some(5)));
    }

    /// The policies under which the sticky override lands as a session
    /// rule: each places a single request by nothing its key hands it. The
    /// gateway's answers with the override are recorded under the first
    /// two.
    const PLACED: [&str; 7] = [
        "random",
        "power_of_two",
        "round_robin",
        "least_load",
        "passthrough",
        "bucket",
        "cache_aware",
    ];

    /// The policy of a preset's name at its flags' defaults, under the
    /// sticky override.
    fn sticky(policy: &str) -> RoutingSettings {
        let mut sticky = config(at_flag_defaults(policy));
        sticky.routing_key_override.enabled = true;
        sticky
    }

    /// Beside a policy that reads no key the override lands as a session
    /// rule on the policy's own binding: the gateway's sticky map in the
    /// document's words. The key is the configured headers, then the
    /// request's id, each without its turn and retry suffixes, and the view
    /// lists the headers, so the rule's cap reads the key's text. The
    /// controller's check takes the document, and its engine asks its host
    /// for the count of a key's requests on a worker.
    #[test]
    fn the_override_lands_as_a_session_rule_beside_a_policy_that_reads_no_key() {
        for name in PLACED {
            let expected = edited(preset(name), |document| {
                document["view"]["headers"] = json!(["x-smg-routing-key"]);
                let binding = &mut document["bindings"][0];
                binding["routing_key"] = json!({
                    "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                    "strip_lineage": true
                });
                binding["sessions"] = json!({
                    "ttl_s": 14_400,
                    "sweep_s": 60,
                    "assignment": "delegate",
                    "inflight_cap": 2,
                    "recovery": "latest",
                    "rule_placed": "untold"
                });
            });
            assert_eq!(sticky(name).policy_document().as_ref(), Ok(&expected));
            let applied = validated(&expected).expect("the controller takes the document");
            assert!(applied.engine().needs().session_in_flight, "{name}");
        }
    }

    /// Beside the prefix ring the override lands as the same session rule
    /// over the same sources: the configured headers, then the request's
    /// own id, each without its turn and retry suffixes. The ring reads a
    /// key too. A header's key leads the prompt, as under the policy alone:
    /// the gateway hands its policy that key without its suffixes
    /// (`model_gateway/src/policies/registry.rs:646`). The key the
    /// request's own id hands names the session alone: the gateway's policy
    /// never hashes it (`model_gateway/src/policies/prefix_hash.rs:384`).
    /// With no header configured the id is the one source.
    #[test]
    fn the_override_lands_as_a_session_rule_beside_the_prefix_ring() {
        let expected = edited(preset("prefix_hash"), |document| {
            document["view"]["headers"] = json!(["x-smg-routing-key"]);
            let binding = &mut document["bindings"][0];
            binding["routing_key"] = json!({
                "sources": [{ "header": "x-smg-routing-key" }, { "rid": true }],
                "strip_lineage": true
            });
            binding["sessions"] = json!({
                "ttl_s": 14_400,
                "sweep_s": 60,
                "assignment": "delegate",
                "inflight_cap": 2,
                "recovery": "latest",
                "rule_placed": "untold"
            });
        });
        let settings = sticky("prefix_hash");
        assert_eq!(settings.policy_document().as_ref(), Ok(&expected));
        let applied = validated(&expected).expect("the controller takes the document");
        assert!(applied.engine().needs().session_in_flight);

        let mut bare = sticky("prefix_hash");
        bare.routing_key_override.headers.clear();
        let document = bare.policy_document().expect("a document");
        assert!(validated(&document).is_ok(), "{document:?}");
        let binding = document.bindings.into_iter().next().expect("the binding");
        let key = binding.routing_key.expect("the key's sources");
        assert_eq!(key.sources.len(), 1);
        assert!(key.sources[0].rid && key.strip_lineage);
        assert!(document.view.headers.is_empty());
    }

    /// The override's rule stands in the way of one pick. `passthrough`
    /// picks alone, so the document's answer for its place is one pick
    /// under the settings as they default. Under the override the answer
    /// names the rule: no caller is told to pick by itself past a pinned
    /// key.
    #[test]
    fn the_overrides_rule_keeps_a_place_from_one_pick() {
        let place_of = |settings: RoutingSettings| {
            let document = settings.policy_document().expect("a document");
            let applied = validated(&document).expect("the controller takes the document");
            let place = applied.engine().place_for("", Phase::Single);
            let ruled = Place::NotAlone {
                profile: "passthrough",
                in_the_way: InTheWay::Sessions,
            };
            (matches!(place, Place::Alone(_)), place == ruled)
        };
        let plain = config(at_flag_defaults("passthrough"));
        assert_eq!(place_of(plain), (true, false));
        assert_eq!(place_of(sticky("passthrough")), (false, true));
    }

    /// The override's own settings land in its rule. A key's first worker
    /// is drawn by the assignment, and the pins' order follows it: the
    /// latest worker leads under `delegate`, the first under the sticky
    /// map's own three. Either time at zero pins a key for ever. Each
    /// configured header is a source, lowercase and once, ahead of the
    /// request's id, and with none configured the id is the one source.
    #[test]
    fn the_overrides_settings_land_in_its_session_rule() {
        let written = |set: &dyn Fn(&mut RoutingKeyOverrideSettings)| {
            let mut sticky = sticky("random");
            set(&mut sticky.routing_key_override);
            let document = sticky.policy_document().expect("a document");
            assert!(validated(&document).is_ok(), "{document:?}");
            let binding = document.bindings.into_iter().next().expect("the binding");
            let rule = binding.sessions.expect("the override's rule");
            let key = binding.routing_key.expect("the key's sources");
            (rule, key, document.view.headers)
        };
        for (assignment, recovery) in [
            (Assignment::Delegate, Recovery::Latest),
            (Assignment::Random, Recovery::Original),
            (Assignment::MinLoad, Recovery::Original),
            (Assignment::MinGroup, Recovery::Original),
        ] {
            let (rule, ..) = written(&|sticky| sticky.assignment_mode = assignment);
            assert_eq!((rule.assignment, rule.recovery), (assignment, recovery));
            // The rule reads nothing of the policy beside it.
            let mut other = sticky("power_of_two");
            other.routing_key_override.assignment_mode = assignment;
            let document = other.policy_document().expect("a document");
            assert!(validated(&document).is_ok(), "{document:?}");
            let beside = document.bindings[0].sessions.as_ref();
            assert_eq!(beside, Some(&rule), "{assignment:?} beside power_of_two");
        }

        let times = |idle, interval| {
            let (rule, ..) = written(&|sticky| {
                sticky.max_idle_secs = idle;
                sticky.eviction_interval_secs = interval;
            });
            (rule.ttl_s, rule.sweep_s)
        };
        assert_eq!(times(600, 30), (600, 30));
        assert_eq!(times(0, 30), (u32::MAX, u32::MAX));
        assert_eq!(times(600, 0), (u32::MAX, u32::MAX));
        assert_eq!(times(0, 0), (u32::MAX, u32::MAX));
        let mut past = sticky("random");
        past.routing_key_override.max_idle_secs = u64::from(u32::MAX);
        assert_eq!(
            refusal(&past),
            ("routing_key_override.max_idle_secs", u32::MAX.to_string())
        );
        past.routing_key_override.max_idle_secs = 600;
        past.routing_key_override.eviction_interval_secs = u64::from(u32::MAX);
        assert_eq!(
            refusal(&past).0,
            "routing_key_override.eviction_interval_secs"
        );

        let sources = |headers: &[&str]| {
            let (_, key, view) = written(&|sticky| {
                sticky.headers = headers.iter().map(|name| (*name).to_string()).collect();
            });
            assert!(key.strip_lineage);
            let read: Vec<&str> = key
                .sources
                .iter()
                .map(|source| source.header.as_deref().unwrap_or("rid"))
                .collect();
            (read.join(" "), view.join(" "))
        };
        assert_eq!(
            sources(&["X-Session-Id", "x-smg-routing-key", "x-session-id"]),
            (
                "x-session-id x-smg-routing-key rid".to_string(),
                "x-session-id x-smg-routing-key".to_string()
            )
        );
        assert_eq!(sources(&[]), ("rid".to_string(), String::new()));
    }

    /// The override's rule places single requests alone. A pair plan and a
    /// gateway of many routers are each refused by the setting's name, in
    /// a sentence that says which of the two it is, under every policy the
    /// rule lands beside: the prefix ring among them.
    #[test]
    fn the_override_is_refused_where_its_rule_has_no_document() {
        let why = |config: &RoutingSettings| match config.policy_document() {
            Err(RoutingSettingsError::UnsupportedSetting {
                setting: "routing_key_override.enabled",
                reason,
                ..
            }) => reason,
            other => panic!("expected the override refused by name, got {other:?}"),
        };
        for name in PLACED.into_iter().chain(["prefix_hash"]) {
            let pair = RoutingSettings {
                mode: pair_mode(None, None),
                ..sticky(name)
            };
            assert_eq!(
                why(&pair),
                "the sticky sessions' rule places single requests alone, and this plan places a \
                 pair",
                "{name}"
            );
            assert_eq!(
                why(&many_routers(sticky(name))),
                "the sticky sessions' rule places single requests alone, and a gateway of many \
                 routers serves pairs beside them",
                "{name}"
            );
        }
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
                Ok(with_single(expected, preset("round_robin"))),
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

    /// No document carries an overload threshold: one off its default writes
    /// nothing, alone and beside the shed word. The switch and the shed word
    /// land on the binding, each only off its default.
    #[test]
    fn an_overload_threshold_lands_on_no_binding_and_the_switch_and_the_shed_word_do() {
        let raised = RoutingSettings {
            worker_overload_waiting_requests: Some(32),
            worker_overload_token_usage: Some(0.9),
            ..config(PolicySettings::RoundRobin)
        };
        assert_eq!(raised.policy_document(), Ok(preset("round_robin")));
        let shedding = RoutingSettings {
            worker_overload_shed: true,
            ..raised
        };
        assert_eq!(
            shedding.policy_document(),
            Ok(edited(preset("round_robin"), |document| {
                document["bindings"][0]["overload"] = json!({ "shed_when_all_vetoed": true });
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

    /// A threshold is the gateway's own beside a supplied document too: one
    /// off its default says no other document than the settings without it,
    /// so the supplied document is in force wherever it is without that
    /// threshold.
    #[test]
    fn an_overload_threshold_beside_a_given_document_is_no_disagreement() {
        let given = preset("least_load");
        for settings in [defaults(), config(at_flag_defaults("least_load"))] {
            let raised = RoutingSettings {
                policy_spec: Some(given.clone()),
                worker_overload_waiting_requests: Some(32),
                worker_overload_token_usage: Some(0.9),
                ..settings
            };
            assert_eq!(
                raised.policy_document_in_force(&defaults()),
                Ok(given.clone())
            );
        }
    }

    /// The two thresholds are no part of the comparison with a supplied
    /// document. Beside settings that state nothing else, a document is in
    /// force with one threshold at none and with both at none, whether its
    /// block is on or off: what a block that is on means over no threshold
    /// is its host's to say. The settings' own document still switches the
    /// block off for both at none.
    #[test]
    fn overload_thresholds_at_none_beside_a_given_document_are_no_disagreement() {
        let on = preset("least_load");
        let off = edited(preset("least_load"), |document| {
            document["bindings"][0]["overload"] = json!({ "enabled": false });
        });
        for (waiting, usage) in [(None, Some(0.9)), (Some(32), None), (None, None)] {
            for given in [&on, &off] {
                let beside = RoutingSettings {
                    policy_spec: Some(given.clone()),
                    worker_overload_waiting_requests: waiting,
                    worker_overload_token_usage: usage,
                    ..defaults()
                };
                assert_eq!(
                    beside.policy_document_in_force(&defaults()),
                    Ok(given.clone()),
                    "{waiting:?} and {usage:?}"
                );
            }
        }
        let own = RoutingSettings {
            worker_overload_waiting_requests: None,
            worker_overload_token_usage: None,
            ..defaults()
        };
        // A host whose own defaults state no threshold: the settings' two
        // stand beside a supplied document all the same.
        for (waiting, usage) in [(Some(8), None), (None, Some(0.8)), (Some(8), Some(0.8))] {
            let beside = RoutingSettings {
                policy_spec: Some(on.clone()),
                worker_overload_waiting_requests: waiting,
                worker_overload_token_usage: usage,
                ..defaults()
            };
            assert_eq!(
                beside.policy_document_in_force(&own),
                Ok(on.clone()),
                "{waiting:?} and {usage:?} over a host's defaults at none"
            );
        }
        let block = |document: PolicySpec| {
            document
                .bindings
                .first()
                .and_then(|binding| binding.overload.clone())
        };
        let switched_off = own.policy_document_in_force(&defaults());
        assert_eq!(switched_off, own.policy_document());
        assert_eq!(switched_off.map(block), Ok(block(off)));
    }

    #[test]
    fn a_setting_no_document_says_is_refused_by_its_name() {
        let round_robin = || config(PolicySettings::RoundRobin);
        // The override's rule places single requests alone.
        let paired = RoutingSettings {
            mode: pair_mode(None, None),
            ..sticky("random")
        };
        let mut selection = config(at_flag_defaults("cache_aware"));
        if let PolicySettings::CacheAware {
            selection_policy, ..
        } = &mut selection.policy
        {
            *selection_policy = Some("another".to_string());
        }
        for (refused, setting, value) in [
            (paired, "routing_key_override.enabled", "true"),
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

    /// What a host asks ahead of the document: a provider mode of a gateway
    /// of one router is named, and no other configuration is.
    #[test]
    fn a_provider_mode_of_one_router_is_named_alone() {
        let named = |mode: ModeSettings, enable_igw: bool| {
            RoutingSettings {
                mode,
                enable_igw,
                ..config(PolicySettings::RoundRobin)
            }
            .provider_alone()
        };
        let provider = || ModeSettings::Provider("a_provider");
        assert_eq!(named(provider(), false), Some("a_provider"));
        // Under the multi-router setting the document plans the gateway's
        // self-hosted workers.
        assert_eq!(named(provider(), true), None);
        for many in [false, true] {
            let pair = ModeSettings::PrefillDecode {
                prefill_policy: None,
                decode_policy: None,
            };
            let encode = ModeSettings::EncodePrefillDecode {
                encode_policy: None,
                prefill_policy: None,
                decode_policy: None,
            };
            for mode in [ModeSettings::Regular, pair, encode] {
                assert_eq!(named(mode, many), None);
            }
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

    /// The pair's own document alone, as `presets` plans it: what a pair
    /// mode says where its main policy's single binding has no document.
    fn pair_alone(config: &RoutingSettings) -> Result<PolicySpec, RoutingSettingsError> {
        presets(config).and_then(|planned| written(config, planned))
    }

    /// A document of two families: `pairs`, a pair's own document, bound over
    /// the two phases of a pair, and behind it the binding and the profile
    /// of `single`, the main policy's regular document.
    fn with_single(mut pairs: PolicySpec, single: PolicySpec) -> PolicySpec {
        for binding in &mut pairs.bindings {
            binding.matcher.phase = vec![Phase::Prefill, Phase::Decode];
        }
        pairs.profiles.extend(single.profiles);
        pairs.bindings.extend(single.bindings);
        pairs
    }

    /// A gateway in the pair mode places one single request beside its
    /// pairs: a token count takes one prefill worker, and the gateway's own
    /// routing places it by the main policy. So the mode's document binds
    /// the pair over its two phases and every other request behind it under
    /// the main policy's profile, of one router as of many, and the
    /// controller's check takes it. The pair's own document alone holds no
    /// side for a single request.
    #[test]
    fn a_pair_mode_binds_a_single_request_behind_its_pairs_under_the_main_policy() {
        let settings = round_robin_pairs_under(at_flag_defaults("least_load"));
        let document = settings
            .policy_document()
            .expect("the pair mode's document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["prefill", "decode"] },
                    "plan": "pd-pair",
                    "profiles": { "decode": "decode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "least_load" },
            ]),
        );
        assert_eq!(
            many_routers(settings.clone()).policy_document().as_ref(),
            Ok(&document)
        );

        // Whether a single request has no side, and whether its place is
        // the main policy's profile.
        let single_request = |document: &PolicySpec| {
            let applied = validated(document).expect("the controller takes the document");
            let place = applied.engine().place_for("", Phase::Single);
            let main = Place::NotAlone {
                profile: "least_load",
                in_the_way: InTheWay::Scorer("least-load"),
            };
            (place == Place::NoSide, place == main)
        };
        let alone = pair_alone(&settings).expect("the pair's own document");
        assert_eq!(single_request(&alone), (true, false));
        assert_eq!(single_request(&document), (false, true));
    }

    /// A pair mode whose main policy states a setting no document says
    /// plans its pairs alone: the single binding is left out, of one router
    /// as of many, and a single request has no side there.
    #[test]
    fn a_pair_mode_plans_pairs_alone_where_its_main_policy_has_no_document() {
        // Under the override the hash ring on both sides of the pair has a
        // document, and a main policy of `cache_aware` that names a
        // selection policy has none.
        let mut unsaid = at_flag_defaults("cache_aware");
        if let PolicySettings::CacheAware {
            selection_policy, ..
        } = &mut unsaid
        {
            *selection_policy = Some("another".to_string());
        }
        let mut keyed = RoutingSettings {
            mode: pair_mode(
                Some(PolicySettings::ConsistentHashing),
                Some(PolicySettings::ConsistentHashing),
            ),
            ..config(unsaid)
        };
        keyed.routing_key_override.enabled = true;
        let alone = pair_alone(&keyed).expect("the pair's own document");
        assert_eq!(keyed.policy_document().as_ref(), Ok(&alone));
        assert_eq!(alone.bindings.len(), 1);
        let applied = validated(&alone).expect("the controller takes the document");
        assert_eq!(applied.engine().place_for("", Phase::Single), Place::NoSide);

        // Under a main policy the override has a rule for, the same pair
        // binds its single request, and the rule is the single binding's.
        // The prefix ring is one of them.
        for main in ["random", "prefix_hash"] {
            keyed.policy = at_flag_defaults(main);
            let both = keyed
                .policy_document()
                .expect("the document of two families");
            let rules: Vec<bool> = both
                .bindings
                .iter()
                .map(|binding| binding.sessions.is_some())
                .collect();
            assert_eq!(rules, [false, true], "{main}");
            assert_eq!(validated(&both).err(), None, "{main}");
        }
    }

    /// The view holds one block size, and a prefix ring counts its key in
    /// it without naming one. A main policy that names another size beside a
    /// pair of prefix rings would end their key at another token (blocks of
    /// 96 under a depth of 256) or leave it no whole block (blocks of 512),
    /// and a pair that names the size does the same to a main policy's
    /// ring. Either way a gateway of one router plans the pair alone, as the
    /// pair's own document says it, and the controller's check takes it. A
    /// gateway of many routers serves each family as a fleet of its own and
    /// keeps the two bindings it had. At the ring's own size the single
    /// request is bound.
    #[test]
    fn a_pair_mode_plans_pairs_alone_where_a_block_size_would_move_a_rings_key() {
        let ring = || at_flag_defaults("prefix_hash");
        let both_ways = |blocks: usize| {
            [
                RoutingSettings {
                    mode: pair_mode(Some(ring()), Some(ring())),
                    ..config(cache_aware_over(blocks))
                },
                RoutingSettings {
                    mode: pair_mode(
                        Some(cache_aware_over(blocks)),
                        Some(cache_aware_over(blocks)),
                    ),
                    ..config(ring())
                },
            ]
        };
        for blocks in [96, 512] {
            for settings in both_ways(blocks) {
                let alone = pair_alone(&settings).expect("the pair's own document");
                assert_eq!(validated(&alone).err(), None, "{blocks}");
                assert_eq!(settings.policy_document().as_ref(), Ok(&alone), "{blocks}");
                assert_eq!(
                    many_routers(settings)
                        .policy_document()
                        .map(|both| both.bindings.len()),
                    Ok(2),
                    "{blocks}"
                );
            }
        }
        for settings in both_ways(16) {
            let both = settings
                .policy_document()
                .expect("the document of two families");
            assert_eq!(both.bindings.len(), 2);
            assert_eq!(validated(&both).err(), None);
        }
        // The sides of one binding say one document together: a ring beside
        // a side that names the size counts in it, as it did, and the
        // single request is bound under a main policy that names none.
        let mixed = RoutingSettings {
            mode: pair_mode(Some(ring()), Some(cache_aware_over(96))),
            ..config(PolicySettings::RoundRobin)
        };
        let both = mixed
            .policy_document()
            .expect("the document of two families");
        assert_eq!((both.bindings.len(), both.view.block_size), (2, Some(96)));
        assert_eq!(validated(&both).err(), None);
    }

    /// Over gRPC or ZMQ a gateway in the pair mode counts tokens with no
    /// worker. Its document of one router is then the pair's own: it binds
    /// no single request, and reads nothing for the main policy. A gateway
    /// of many routers serves single workers however it reaches them, and
    /// over HTTP the single request is bound. A given document beside no
    /// routing setting is in force over each.
    #[test]
    fn a_pair_mode_of_one_router_binds_a_single_request_over_http_alone() {
        let over = |connection_mode| RoutingSettings {
            connection_mode,
            ..round_robin_pairs_under(at_flag_defaults("cache_aware"))
        };
        let events = |document: &PolicySpec| {
            validated(document)
                .map(|applied| applied.needs().events)
                .ok()
        };
        let both = over(ConnectionModeSetting::Http)
            .policy_document()
            .expect("the document over HTTP");
        assert_eq!((both.bindings.len(), events(&both)), (2, Some(true)));

        let given = preset("round_robin");
        for connection_mode in [ConnectionModeSetting::Grpc, ConnectionModeSetting::Zmq] {
            let settings = over(connection_mode);
            let alone = pair_alone(&settings).expect("the pair's own document");
            assert_eq!(
                (alone.bindings.len(), events(&alone)),
                (1, Some(false)),
                "{connection_mode:?}"
            );
            assert_eq!(
                settings.policy_document().as_ref(),
                Ok(&alone),
                "{connection_mode:?}"
            );
            assert_eq!(
                many_routers(settings).policy_document().as_ref(),
                Ok(&both),
                "{connection_mode:?}"
            );
            let unset = RoutingSettings {
                mode: pair_mode(None, None),
                connection_mode,
                policy_spec: Some(given.clone()),
                ..defaults()
            };
            assert_eq!(
                unset.policy_document_in_force(&defaults()),
                Ok(given.clone()),
                "{connection_mode:?}"
            );
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
            let paired = pair_alone(&pair).expect("the pair's own document");
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
            // same document, of one router as of many, and the controller
            // takes it.
            assert_eq!(pair.policy_document().as_ref(), Ok(&both), "{name}");
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
        // The block states its shed word and no number: a threshold off its
        // default lands on no binding.
        let sheds = OverloadSpec {
            shed_when_all_vetoed: true,
            ..OverloadSpec::default()
        };
        assert_eq!(overload, Some(sheds));
        assert!(key.is_some());

        let both = many_routers(keyed)
            .policy_document()
            .expect("the document of many routers");
        assert_eq!(both.bindings.len(), 2);
        for binding in &both.bindings {
            // One switch and one shed word stand for the pools of each
            // family, and each family reads a request's key from the same
            // sources.
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

    /// A pair mode of one router says two bindings, so a given document is
    /// compared with both. Beside a stated policy, the pair's own document
    /// is refused at the single request's profile, which it lacks, and the
    /// document of two bindings is in force. Where no routing setting is
    /// set, the pair's own document is in force.
    #[test]
    fn a_given_document_is_compared_with_the_two_bindings_of_a_pair_mode() {
        let stated = round_robin_pairs_under(at_flag_defaults("least_load"));
        let said = stated.policy_document().expect("the mode's document");
        let alone = pair_alone(&stated).expect("the pair's own document");
        assert_eq!((said.bindings.len(), alone.bindings.len()), (2, 1));
        let given = |document: &PolicySpec, settings: &RoutingSettings| RoutingSettings {
            policy_spec: Some(document.clone()),
            ..settings.clone()
        };

        let refused = given(&alone, &stated).policy_document_in_force(&defaults());
        assert!(
            matches!(
                &refused,
                Err(RoutingSettingsError::Disagreement { at, document, .. })
                    if at == "profiles.least_load" && document == "nothing"
            ),
            "{refused:?}"
        );
        assert_eq!(
            given(&said, &stated).policy_document_in_force(&defaults()),
            Ok(said)
        );
        let unset = RoutingSettings {
            mode: pair_mode(None, None),
            ..defaults()
        };
        assert_eq!(
            given(&alone, &unset).policy_document_in_force(&defaults()),
            Ok(alone)
        );
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
        let pairs_alone = pair_alone(&keyed);
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(many_routers(keyed.clone()).policy_document(), pairs_alone);
        // The same main policy alone is refused by the setting's name.
        keyed.mode = ModeSettings::Regular;
        assert_eq!(
            refusal(&many_routers(keyed)),
            ("routing_key_override.enabled", "true".to_string())
        );
    }

    /// A gateway of many routers in the encode mode serves single workers
    /// beside its encoders and pairs: a model of single workers is placed
    /// by the main policy. So its document binds the mode's three sides
    /// over their phases and every other request behind them, in that
    /// order: the first binding that matches a request decides it.
    #[test]
    fn the_encode_mode_of_many_routers_binds_a_single_request_behind_its_sides() {
        let encode_mode = |encode: Option<PolicySettings>, sides: Option<PolicySettings>| {
            ModeSettings::EncodePrefillDecode {
                encode_policy: encode,
                decode_policy: sides.clone(),
                prefill_policy: sides,
            }
        };
        let settings = many_routers(RoutingSettings {
            mode: encode_mode(None, None),
            ..config(PolicySettings::RoundRobin)
        });
        let document = settings.policy_document().expect("the mode's document");
        assert_eq!(
            serde_json::to_value(&document.bindings).expect("a written form"),
            json!([
                {
                    "match": { "model": "*", "phase": ["encode", "prefill", "decode"] },
                    "plan": "epd",
                    "profiles": { "decode": "decode", "encode": "encode", "prefill": "prefill" },
                },
                { "match": { "model": "*", "phase": [] }, "profile": "round_robin" },
            ]),
        );
        let applied = validated(&document).expect("the controller takes the document");
        assert_ne!(applied.engine().place_for("", Phase::Single), Place::NoSide);

        // A prefix ring counts its key in its own binding's blocks. Where
        // the main policy names another size, the mode's own document
        // stands, as it does under one router, and the controller takes it.
        let ringed = RoutingSettings {
            mode: encode_mode(None, Some(at_flag_defaults("prefix_hash"))),
            ..config(cache_aware_over(512))
        };
        let own = ringed.policy_document().expect("the mode's own document");
        assert!(validated(&own).is_ok());
        assert_eq!(many_routers(ringed).policy_document().as_ref(), Ok(&own));

        // A main policy the controller's check refuses is no document of
        // the single family: a prefix ring the view carries no block for
        // (its depth under one block or past the blocks a request carries,
        // or a boundary under one block), or a cache-aware setting the
        // build does not run. The mode's own document stands there too, the
        // controller takes it, and a single request has no side under it.
        let mut mains = Vec::new();
        for (depth, boundaries, alone) in [
            (15, vec![], true),
            (16, vec![], false),
            (4111, vec![], false),
            (4112, vec![], true),
            (256, vec![8], true),
        ] {
            let named = format!("a main ring of {depth} tokens over {boundaries:?}");
            let mut main = at_flag_defaults("prefix_hash");
            if let PolicySettings::PrefixHash {
                prefix_token_count,
                cache_boundaries,
                ..
            } = &mut main
            {
                (*prefix_token_count, *cache_boundaries) = (depth, boundaries);
            }
            mains.push((named, main, alone));
        }
        for setting in [
            "",
            "overlap_decay",
            "selection_temperature",
            "selection_accounting_ttl_ms",
            "balance_uncached_credit",
        ] {
            let mut main = at_flag_defaults("cache_aware");
            if let PolicySettings::CacheAware {
                overlap_decay,
                selection_temperature,
                selection_accounting_ttl_ms,
                balance_uncached_credit,
                ..
            } = &mut main
            {
                match setting {
                    "overlap_decay" => *overlap_decay = 0.5,
                    "selection_temperature" => *selection_temperature = 0.05,
                    "selection_accounting_ttl_ms" => *selection_accounting_ttl_ms = 200,
                    "balance_uncached_credit" => *balance_uncached_credit = true,
                    _ => {}
                }
            }
            let named = format!("a cache-aware main policy that sets {setting:?}");
            mains.push((named, main, !setting.is_empty()));
        }
        for (named, main, alone) in mains {
            let round_robin = Some(PolicySettings::RoundRobin);
            let uncarried = RoutingSettings {
                mode: encode_mode(round_robin.clone(), round_robin),
                ..config(main)
            };
            let own = uncarried
                .policy_document()
                .expect("the mode's own document");
            let said = many_routers(uncarried)
                .policy_document()
                .expect("the settings' document");
            let applied = validated(&said).unwrap_or_else(|refusal| panic!("{named}: {refusal}"));
            let unplaced = applied.engine().place_for("", Phase::Single) == Place::NoSide;
            assert_eq!((said == own, unplaced), (alone, alone), "{named}");
        }

        // No configuration of the mode loses its document to its single
        // request. Every policy is the main one in turn, the encode side
        // and the pair's sides are unnamed or of one policy, and the sticky
        // override is off and on. Where the mode's own document, as
        // `presets` plans it, is refused, the settings' is refused by the
        // same sentence. Where the controller's check takes the mode's
        // own, it takes the settings', which binds a single request behind
        // the mode's sides in all but these: under the override, beside
        // sides of the hash ring, every main policy but
        // `consistent_hashing` and `manual` has no document for its keyed
        // requests, and the mode is planned alone, whatever its encode
        // side.
        let unnamed_or_each = || std::iter::once(None).chain(PRESETS.iter().copied().map(Some));
        let mut alone_still = Vec::new();
        for main in PRESETS {
            for encode in unnamed_or_each() {
                for sides in unnamed_or_each() {
                    for sticky in [false, true] {
                        let named = format!("{main} over {encode:?} and {sides:?}, {sticky}");
                        let mut swept = many_routers(RoutingSettings {
                            mode: encode_mode(
                                encode.map(at_flag_defaults),
                                sides.map(at_flag_defaults),
                            ),
                            ..config(at_flag_defaults(main))
                        });
                        swept.routing_key_override.enabled = sticky;
                        let alone = presets(&swept).and_then(|planned| written(&swept, planned));
                        let said = swept.policy_document();
                        let (Ok(one), Ok(both)) = (&alone, &said) else {
                            assert_eq!(said, alone, "{named}");
                            continue;
                        };
                        let taken = validated(one).is_ok();
                        assert_eq!(validated(both).is_ok(), taken, "{named}");
                        if taken && both.bindings.len() == 1 {
                            assert_eq!(both, one, "{named}");
                            alone_still.push((main, sides, sticky));
                        }
                    }
                }
            }
        }
        alone_still.dedup();
        let ring = Some("consistent_hashing");
        let keyless = [
            "random",
            "round_robin",
            "passthrough",
            "power_of_two",
            "least_load",
            "prefix_hash",
            "cache_aware",
            "bucket",
        ];
        assert_eq!(alone_still, keyless.map(|main| (main, ring, true)));
    }

    /// The labels of these models, each with the preset name its label
    /// resolves to.
    fn labels(named: &[(&str, &str)]) -> BTreeMap<String, String> {
        named
            .iter()
            .map(|&(model, policy)| (model.to_owned(), policy.to_owned()))
            .collect()
    }

    /// The settings' document with these labels.
    fn with_labels(
        settings: &RoutingSettings,
        named: &[(&str, &str)],
    ) -> Result<PolicySpec, RoutingSettingsError> {
        settings.policy_document_with_labels(&labels(named))
    }

    /// A label's refusal, in its three parts.
    fn label_error(model: &str, policy: &str, reason: &str) -> RoutingSettingsError {
        RoutingSettingsError::Label {
            model: model.to_owned(),
            policy: policy.to_owned(),
            reason: reason.to_owned(),
        }
    }

    /// A main policy that is not the label's: round robin, and `random`
    /// beside a label of round robin.
    fn main_beside(label: &str) -> PolicySettings {
        at_flag_defaults(if label == "round_robin" {
            "random"
        } else {
            "round_robin"
        })
    }

    /// The profile the document binds a single request of `model` to, as
    /// the controller's check builds the document.
    fn single_profile(document: &PolicySpec, model: &str) -> String {
        let engine = document
            .validate_with(&["x-smg-tenant-id"])
            .expect("the controller's check takes the document");
        let binding = engine
            .binding(model, Phase::Single)
            .expect("a binding for the model's single request");
        engine.profiles()[usize::from(binding.profile)].name.clone()
    }

    /// The index the document answers for `model`: its most specific entry,
    /// or the defaults where no entry names the model.
    fn index_of(document: &PolicySpec, model: &str) -> IndexSpec {
        let entry = document.index_for(model);
        entry.map_or_else(IndexSpec::default, |(_, index)| index.clone())
    }

    /// A label builds each policy as the gateway's registry builds it from
    /// a name alone. Eight of the ten are the policy at its flags' defaults,
    /// the preset. `cache_aware` and `bucket` are not: seven numbers are the
    /// registry's own, and tidying them to the presets' turns this case red.
    /// A name's spelling is its host's to resolve: only a preset's name
    /// names a policy here.
    #[test]
    fn a_label_builds_two_policies_at_other_numbers_than_their_presets() {
        for name in PRESETS {
            let by_label = PolicySettings::of_label(name).expect("a policy of each preset's name");
            assert_eq!(by_label.name(), name);
            let preset = by_label == at_flag_defaults(name);
            assert_eq!(preset, !["cache_aware", "bucket"].contains(&name), "{name}");
        }
        assert_eq!(
            PolicySettings::of_label("cache_aware"),
            Some(cache_aware(0.5, 32, 1.1, 30, 10_000))
        );
        assert_eq!(
            PolicySettings::of_label("bucket"),
            Some(PolicySettings::Bucket {
                balance_abs_threshold: 32,
                balance_rel_threshold: 1.0001,
                bucket_adjust_interval_secs: 5,
            })
        );
        for unnamed in ["cacheaware", "Bucket", "least_loaded", ""] {
            assert_eq!(PolicySettings::of_label(unnamed), None, "{unnamed:?}");
        }
        // A label's `cache_aware` names the block size a silent ring counts
        // in: beside it no ring of the settings' sides loses its blocks.
        let silent = usize::try_from(Gates::default().block_size).expect("a block size");
        assert!(matches!(
            PolicySettings::of_label("cache_aware"),
            Some(PolicySettings::CacheAware { block_size, .. }) if block_size == silent
        ));
    }

    /// With no label the labelled document is the settings' own, and a
    /// label that names the main policy changes nothing: its model already
    /// runs that profile, with the settings' options.
    #[test]
    fn no_label_and_a_label_of_the_main_policy_leave_the_settings_document() {
        for name in PRESETS {
            for settings in [
                config(at_flag_defaults(name)),
                round_robin_pairs_under(at_flag_defaults(name)),
                many_routers(config(at_flag_defaults(name))),
            ] {
                let own = settings.policy_document();
                assert_eq!(with_labels(&settings, &[]), own, "{name}");
                assert_eq!(with_labels(&settings, &[("m", name)]), own, "{name}");
            }
        }
    }

    /// Each stock policy as a label, beside another main policy: the
    /// label's binding stands ahead of the settings' own, which stays as it
    /// was with its profile. The controller's check takes the document, and
    /// its engine binds a single request of the labelled model to the
    /// label's profile and another model's to the main policy's.
    #[test]
    fn each_stock_policy_as_a_label_binds_its_model_ahead_of_every_model() {
        for name in PRESETS {
            let settings = config(main_beside(name));
            let main = settings.policy.name();
            let own = settings.policy_document().expect("the settings' document");
            let document = with_labels(&settings, &[("labelled", name)]).expect("a labelled one");
            assert_eq!(document.bindings.len(), 2, "{name}");
            assert_eq!(document.bindings[0].matcher.model, "labelled", "{name}");
            assert!(document.bindings[0].matcher.phase.is_empty(), "{name}");
            assert_eq!(document.bindings[0].profile.as_deref(), Some(name));
            assert_eq!(document.bindings[1], own.bindings[0], "{name}");
            assert_eq!(document.profiles[main], own.profiles[main], "{name}");
            assert_eq!(single_profile(&document, "labelled"), name);
            assert_eq!(single_profile(&document, "another"), main);
        }
    }

    /// One way to set a configuration, by its name in a case's messages.
    type Way<'a> = (&'a str, &'a dyn Fn(&mut RoutingSettings));

    /// In a regular gateway of one router, a label's binding is the one its
    /// policy would get as the main policy of the same settings: its own
    /// profile, key sources and session rule, by the code that writes a main
    /// policy's, and none of the main binding's. So the warm-up settings
    /// land in a labelled `cache_aware` profile, configured headers lead a
    /// keyed label's sources, and under the sticky override a label carries
    /// the override's rule where the override places its policy. Where the
    /// policy has no document as the main policy there, the label is refused
    /// by that sentence. A mode whose side takes the main policy is no such
    /// comparison: as the main policy the label's would place that side too.
    #[test]
    fn a_labels_binding_is_the_one_its_policy_gets_as_the_main_policy() {
        let warm = |settings: &mut RoutingSettings| settings.worker_warmup_secs = 7;
        let headed = |settings: &mut RoutingSettings| {
            settings.routing_key_override.headers = vec!["x-session".to_owned()];
        };
        let sticky = |settings: &mut RoutingSettings| settings.routing_key_override.enabled = true;
        let both = |settings: &mut RoutingSettings| {
            settings.routing_key_override.enabled = true;
            settings.routing_key_override.headers = vec!["x-session".to_owned()];
        };
        let plain = |_: &mut RoutingSettings| {};
        let unprotected =
            |settings: &mut RoutingSettings| settings.worker_overload_protection = false;
        let ways: [Way<'_>; 6] = [
            ("as set by none", &plain),
            ("the overload protection off", &unprotected),
            ("a warm-up of seven seconds", &warm),
            ("a configured header", &headed),
            ("the sticky override", &sticky),
            ("the override over a configured header", &both),
        ];
        let mut refused = Vec::new();
        for name in PRESETS {
            for (way, set) in ways {
                let mut settings = config(main_beside(name));
                set(&mut settings);
                let main = settings.policy.name();
                let own = settings
                    .policy_document()
                    .expect("the main policy's document");
                let as_main = RoutingSettings {
                    policy: PolicySettings::of_label(name).expect("the label's policy"),
                    ..settings.clone()
                }
                .policy_document();
                let said = with_labels(&settings, &[("m", name)]);
                let as_main = match as_main {
                    Ok(document) => document,
                    Err(refusal) => {
                        let expected = label_error("m", name, &refusal.to_string());
                        assert_eq!(said, Err(expected), "{name}, {way}");
                        refused.push((name, way));
                        continue;
                    }
                };
                let document = said.unwrap_or_else(|refusal| panic!("{name}, {way}: {refusal}"));
                let mut expected = as_main.bindings[0].clone();
                expected.matcher.model = "m".to_owned();
                assert_eq!(document.bindings[0], expected, "{name}, {way}");
                assert_eq!(document.bindings[1], own.bindings[0], "{name}, {way}");
                assert_eq!(
                    document.profiles[name], as_main.profiles[name],
                    "{name}, {way}"
                );
                assert_eq!(document.profiles[main], own.profiles[main], "{name}, {way}");
                // The label's model reads the index its policy gets as the
                // main policy, and every other model the settings' own.
                let index = index_of(&document, "m");
                assert_eq!(index, index_of(&as_main, "m"), "{name}, {way}");
                let index = index_of(&document, "another");
                assert_eq!(index, index_of(&own, "another"), "{name}, {way}");
                // The view's headers: the main binding's, then the label's.
                let mut headers = own.view.headers.clone();
                for header in &as_main.view.headers {
                    if !headers.contains(header) {
                        headers.push(header.clone());
                    }
                }
                assert_eq!(document.view.headers, headers, "{name}, {way}");
                assert!(validated(&document).is_ok(), "{name}, {way}");
            }
        }
        // The settings' overload block stands on a label's binding.
        let mut exposed = config(main_beside("random"));
        unprotected(&mut exposed);
        let document = with_labels(&exposed, &[("m", "random")]).expect("a labelled one");
        let switched_off = Some(OverloadSpec {
            enabled: false,
            ..OverloadSpec::default()
        });
        assert_eq!(document.bindings[0].overload, switched_off);
        assert_eq!(document.bindings[1].overload, switched_off);
        // The warm-up setting is in the labelled profile, not the preset's.
        let mut warmed = config(main_beside("cache_aware"));
        warm(&mut warmed);
        let cold = config(main_beside("cache_aware"));
        let profile = |settings: &RoutingSettings| {
            let document = with_labels(settings, &[("m", "cache_aware")]).expect("a labelled one");
            document.profiles["cache_aware"].clone()
        };
        assert_ne!(profile(&warmed), profile(&cold));
        // Under the override a policy that places by the key itself reads
        // the routing key header as sent, behind a configured header.
        assert_eq!(
            refused,
            [
                (
                    "consistent_hashing",
                    "the override over a configured header"
                ),
                ("manual", "the override over a configured header"),
            ]
        );
    }

    /// The session rule and the key sources are each policy's own, either
    /// way round. Under a main policy of `manual` a round robin label's
    /// binding names no rule and no key, so its model's keyed requests take
    /// turns. Under round robin a `manual` label brings the manual rule and
    /// its header to the view, and a hash ring label its own sources.
    #[test]
    fn a_label_reads_its_own_key_and_rule_and_not_the_main_bindings() {
        let manual = config(at_flag_defaults("manual"));
        let own = manual.policy_document().expect("the manual document");
        let document = with_labels(&manual, &[("m", "round_robin")]).expect("a labelled one");
        assert_eq!(document.bindings[0].sessions, None);
        assert_eq!(document.bindings[0].routing_key, None);
        assert!(own.bindings[0].sessions.is_some());
        assert_eq!(document.bindings[1], own.bindings[0]);
        assert_eq!(document.view, own.view);

        let turns = config(at_flag_defaults("round_robin"));
        let own = turns.policy_document().expect("the round robin document");
        assert!(own.view.headers.is_empty());
        let document = with_labels(&turns, &[("m", "manual")]).expect("a labelled one");
        assert_eq!(
            document.bindings[0].sessions,
            preset("manual").bindings[0].sessions
        );
        assert_eq!(document.view.headers, ["x-smg-routing-key"]);
        assert_eq!(document.bindings[1], own.bindings[0]);
        let document = with_labels(&turns, &[("m", "consistent_hashing")]).expect("a labelled one");
        assert_eq!(
            document.bindings[0].routing_key,
            preset("consistent_hashing").bindings[0].routing_key
        );
        assert_eq!(document.bindings[0].sessions, None);
        assert_eq!(document.view, own.view);
    }

    /// A pair mode keeps its pair's binding under a label: the label's
    /// stands behind it and ahead of the binding of every model's single
    /// requests. The engine binds a prefill of the labelled model to the
    /// pair plan and its single request to the label's profile.
    #[test]
    fn a_pair_mode_keeps_its_pair_binding_ahead_of_a_labels() {
        for settings in [
            round_robin_pairs_under(at_flag_defaults("least_load")),
            many_routers(config(at_flag_defaults("least_load"))),
        ] {
            let own = settings
                .policy_document()
                .expect("a document of two families");
            assert_eq!(own.bindings.len(), 2);
            let document = with_labels(&settings, &[("m", "random")]).expect("a labelled one");
            assert_eq!(document.bindings.len(), 3);
            assert_eq!(document.bindings[0], own.bindings[0]);
            assert_eq!(document.bindings[1].matcher.model, "m");
            assert_eq!(document.bindings[1].profile.as_deref(), Some("random"));
            assert_eq!(document.bindings[1].overload, own.bindings[1].overload);
            assert_eq!(document.bindings[2], own.bindings[1]);
            let engine = document
                .validate_with(&["x-smg-tenant-id"])
                .expect("the controller's check takes the document");
            let plan = |phase| engine.binding("m", phase).map(|binding| binding.plan);
            assert_eq!(plan(Phase::Prefill), Some(Plan::PdPair));
            assert_eq!(plan(Phase::Decode), Some(Plan::PdPair));
            assert_eq!(plan(Phase::Single), Some(Plan::Single));
            assert_eq!(single_profile(&document, "m"), "random");
            assert_eq!(single_profile(&document, "another"), "least_load");
        }
    }

    /// Where the settings' document binds no single request, no label is
    /// bound and the model's single requests stay refused: a pair mode of
    /// one router over gRPC, the encode mode of one router, and a pair
    /// planned alone for a main policy's setting no document says.
    #[test]
    fn a_label_adds_no_binding_where_no_single_request_is_bound() {
        let pairs = || round_robin_pairs_under(at_flag_defaults("least_load"));
        let mut unsaid = at_flag_defaults("cache_aware");
        if let PolicySettings::CacheAware {
            selection_policy, ..
        } = &mut unsaid
        {
            *selection_policy = Some("another".to_owned());
        }
        for (settings, sides) in [
            (
                RoutingSettings {
                    connection_mode: ConnectionModeSetting::Grpc,
                    ..pairs()
                },
                2,
            ),
            (
                RoutingSettings {
                    mode: ModeSettings::EncodePrefillDecode {
                        encode_policy: None,
                        prefill_policy: None,
                        decode_policy: None,
                    },
                    ..config(at_flag_defaults("round_robin"))
                },
                3,
            ),
            (round_robin_pairs_under(unsaid), 2),
        ] {
            let own = settings
                .policy_document()
                .expect("a document of the mode alone");
            assert_eq!(own.bindings.len(), 1);
            assert_eq!(own.bindings[0].profiles.len(), sides);
            assert_eq!(with_labels(&settings, &[("m", "random")]), Ok(own));
            // A label's own id and name are checked here as everywhere.
            assert_eq!(
                with_labels(&settings, &[("m", "none")]),
                Err(label_error(
                    "m",
                    "none",
                    &SpecError::UnknownPreset("none".to_owned()).to_string()
                ))
            );
            assert!(matches!(
                with_labels(&settings, &[("m*", "random")]),
                Err(RoutingSettingsError::Label { model, .. }) if model == "m*"
            ));
        }
        // What a label's binding would cost the document is checked only
        // where the binding is written: a ring another block size refuses
        // beside a single binding is passed over where there is none.
        let blocks = round_robin_pairs_under(cache_aware_over(32));
        assert!(matches!(
            with_labels(&blocks, &[("m", "prefix_hash")]),
            Err(RoutingSettingsError::Label { reason, .. }) if reason.starts_with("block_size = 16 and 32: ")
        ));
        let unbound = RoutingSettings {
            connection_mode: ConnectionModeSetting::Grpc,
            ..blocks
        };
        let own = unbound.policy_document().expect("the pair's document");
        assert_eq!(own.bindings.len(), 1);
        assert_eq!(with_labels(&unbound, &[("m", "prefix_hash")]), Ok(own));
    }

    /// Settings no document says are refused as before, whatever the
    /// labels: the settings' own refusal comes first.
    #[test]
    fn settings_no_document_says_are_refused_as_before_under_a_label() {
        let rank = RoutingSettings {
            dp_minimum_tokens_scheduler: true,
            ..config(at_flag_defaults("round_robin"))
        };
        let mut sticky_ring = config(at_flag_defaults("consistent_hashing"));
        sticky_ring.routing_key_override.enabled = true;
        sticky_ring.routing_key_override.headers = vec!["x-session".to_owned()];
        for settings in [rank, sticky_ring] {
            let refusal = settings.policy_document().expect_err("no document");
            for named in [
                &[("m", "random")][..],
                &[("m*", "random")],
                &[("m", "none")],
            ] {
                assert_eq!(with_labels(&settings, named).as_ref(), Err(&refusal));
            }
        }
    }

    /// The builder's refusal of a setting under a label's policy is the
    /// label's, by its model and its policy. A parting of the presets and
    /// this module is not: it passes as it is, so a host that leaves a
    /// refused label out does not leave it unsaid.
    #[test]
    fn a_parting_of_the_presets_is_no_labels_refusal() {
        let policy = PolicySettings::Random;
        let parted = || {
            [
                RoutingSettingsError::NoStage {
                    policy: "random",
                    stage: "picker",
                },
                RoutingSettingsError::NoPart {
                    policy: "random",
                    part: "profile",
                },
                RoutingSettingsError::Document(SpecError::UnknownPreset("random".to_owned())),
            ]
        };
        for (passed, parted) in parted().into_iter().zip(parted()) {
            assert_eq!(label_refused("m", &policy, passed), parted);
        }
        let setting = || unsupported("worker_warmup_secs", 7, "no document says it");
        assert_eq!(
            label_refused("m", &policy, setting()),
            label_error("m", "random", &setting().to_string())
        );
    }

    /// A label the document has no binding for is refused by its model and
    /// its policy: a model's id that holds a `*` or is empty, and a name
    /// with no preset. A label of the main policy adds no binding, so its
    /// model's id refuses nothing. That label is the main policy's name as
    /// a preset spells it: another letter case is a name with no preset.
    #[test]
    fn a_label_is_refused_by_its_model_and_its_policy() {
        let settings = config(at_flag_defaults("round_robin"));
        let own = settings.policy_document();
        let pattern = "a binding reads a `*` at the start or the end of a model's name as a pattern and the document refuses one anywhere else, so no binding names this model alone";
        for model in ["*", "m*", "*m", "a*b"] {
            assert_eq!(
                with_labels(&settings, &[(model, "random")]),
                Err(label_error(model, "random", pattern)),
            );
            assert_eq!(with_labels(&settings, &[(model, "round_robin")]), own);
        }
        assert_eq!(
            with_labels(&settings, &[("", "random")]),
            Err(label_error(
                "",
                "random",
                "a binding names its model by the model's id, and this id is empty"
            )),
        );
        assert_eq!(with_labels(&settings, &[("", "round_robin")]), own);
        let unknown = |name: &str| SpecError::UnknownPreset(name.to_owned()).to_string();
        assert_eq!(
            with_labels(&settings, &[("a", "random"), ("m", "least_loaded")]),
            Err(label_error("m", "least_loaded", &unknown("least_loaded"))),
        );
        assert_eq!(
            with_labels(&settings, &[("m", "Round_Robin")]),
            Err(label_error("m", "Round_Robin", &unknown("Round_Robin"))),
        );
        // Every label's name and id is read before a side is written, so
        // the refusal need not be the first label's in the ids' order.
        let blocks = config(cache_aware_over(32));
        assert!(with_labels(&blocks, &[("a", "prefix_hash")]).is_err());
        assert_eq!(
            with_labels(&blocks, &[("a", "prefix_hash"), ("z", "least_loaded")]),
            Err(label_error("z", "least_loaded", &unknown("least_loaded"))),
        );
    }

    /// A label's policy that cannot stand beside the settings' own is
    /// refused as two sides of the settings are, by the builder's own
    /// sentence. The settings' document keeps its families: the refusal is
    /// the label's, and no family is left out for it. A prefix ring keeps
    /// the blocks of its own document, and the builder holds one list of
    /// cache boundaries for the document.
    ///
    /// A label's ring is held to the document's block size in a gateway of
    /// many routers too, where the settings' own document stands as it
    /// stood. Unrefused, that label's ring would have no whole block. The
    /// controller's check would then refuse the document of every model,
    /// with no label named.
    #[test]
    fn a_label_that_cannot_stand_beside_the_settings_is_refused_by_name() {
        let blocks = config(cache_aware_over(32));
        assert_eq!(
            with_labels(&blocks, &[("m", "prefix_hash")]),
            Err(label_error(
                "m",
                "prefix_hash",
                "block_size = 16 and 32: a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own"
            )),
        );
        let fleet = many_routers(config(cache_aware_over(512)));
        let own = fleet.policy_document().expect("the fleet's own document");
        assert_eq!(own.bindings.len(), 2);
        assert_eq!(
            with_labels(&fleet, &[("m", "prefix_hash")]),
            Err(label_error(
                "m",
                "prefix_hash",
                "block_size = 16 and 512: a prefix ring counts its key in the document's one block size, and another binding's policy names another size than the ring's own"
            )),
        );
        assert_eq!(
            with_labels(&fleet, &[("m", "random")]).map(|document| document.bindings.len()),
            Ok(3)
        );

        let mut cut = at_flag_defaults("prefix_hash");
        if let PolicySettings::PrefixHash {
            cache_boundaries, ..
        } = &mut cut
        {
            *cache_boundaries = vec![64];
        }
        let cut = config(cut);
        assert!(cut.policy_document().is_ok());
        match with_labels(&cut, &[("m", "cache_aware")]) {
            Err(RoutingSettingsError::Label {
                model,
                policy,
                reason,
            }) => {
                assert_eq!((model.as_str(), policy.as_str()), ("m", "cache_aware"));
                assert!(reason.starts_with("cache_boundaries = "), "{reason}");
            }
            other => panic!("expected the label refused by its boundaries, got {other:?}"),
        }
    }

    /// A label's index stands under its model's own id, where it says
    /// another thing than the document answers for that model without it.
    /// So a `cache_aware` label builds beside a pair's `cache_aware` side
    /// that names another index, and each model reads its own. Two sides of
    /// the settings that name two indexes are refused, as before.
    ///
    /// A label whose index is the one the settings wrote adds no entry, and
    /// neither does a label of a policy that names no index. Under settings
    /// that write none, each `cache_aware` label writes its model's entry.
    #[test]
    fn a_labels_index_stands_under_its_models_id() {
        let of_label = IndexSpec {
            max_size: 10_000,
            sweep_s: 30,
            ..IndexSpec::default()
        };
        let indexed =
            |prefill: Option<PolicySettings>, decode: Option<PolicySettings>| RoutingSettings {
                mode: pair_mode(prefill, decode),
                ..config(at_flag_defaults("round_robin"))
            };
        let at_flags = || Some(at_flag_defaults("cache_aware"));
        let as_a_label = || PolicySettings::of_label("cache_aware");
        let two_sides = indexed(at_flags(), as_a_label()).policy_document();
        assert!(two_sides.is_err(), "two sides of two indexes");

        let settings = indexed(at_flags(), None);
        let own = settings.policy_document().expect("the settings' document");
        assert_ne!(
            index_of(&own, "m"),
            of_label,
            "the pair's side names another"
        );
        let labels = [("m", "cache_aware"), ("n", "random")];
        let document = with_labels(&settings, &labels).expect("each model its own index");
        assert_eq!(document.index.get("m"), Some(&of_label));
        assert_eq!(document.index.get("n"), None, "`random` names no index");
        assert_eq!(document.index.get("*"), own.index.get("*"));
        assert_eq!(document.index.len(), own.index.len() + 1);
        assert_eq!(index_of(&document, "m"), of_label);
        assert_eq!(index_of(&document, "n"), index_of(&own, "n"));

        let same = indexed(as_a_label(), None);
        let own = same.policy_document().expect("the settings' document");
        assert_eq!(
            index_of(&own, "m"),
            of_label,
            "the settings wrote the label's"
        );
        let document = with_labels(&same, &[("m", "cache_aware")]).expect("a labelled one");
        assert_eq!(document.index, own.index);

        let none = config(at_flag_defaults("round_robin"));
        let labels = [("m", "cache_aware"), ("n", "cache_aware")];
        let document = with_labels(&none, &labels).expect("two labelled ones");
        assert_eq!(
            document.index.into_iter().collect::<Vec<_>>(),
            [
                ("m".to_owned(), of_label.clone()),
                ("n".to_owned(), of_label)
            ]
        );
    }

    /// A refused label fails alone. Two labels never refuse one another, so
    /// each stands or falls beside the settings alone: the refusal names the
    /// one that falls, and every subset of the labels that build builds. A
    /// subset's document holds its labels' bindings as the whole one holds
    /// them, in their order, ahead of the settings' own, and no other.
    #[test]
    fn labels_stand_or_fall_alone_beside_the_settings() {
        let settings = config(cache_aware_over(32));
        let named = [
            ("a", "random"),
            ("b", "prefix_hash"),
            ("c", "manual"),
            ("d", "cache_aware"),
            ("e", "bucket"),
        ];
        let falls = |label: &(&str, &str)| with_labels(&settings, &[*label]).is_err();
        let fallen: Vec<_> = named.iter().filter(|label| falls(label)).collect();
        assert_eq!(fallen, [&("b", "prefix_hash")]);
        match with_labels(&settings, &named) {
            Err(RoutingSettingsError::Label { model, policy, .. }) => {
                assert_eq!((model.as_str(), policy.as_str()), ("b", "prefix_hash"));
            }
            other => panic!("expected the ring's label refused, got {other:?}"),
        }
        // The main policy is `cache_aware`: its own label adds nothing.
        let built = [named[0], named[2], named[3], named[4]];
        let whole = with_labels(&settings, &built).expect("the labels that build");
        let models = |document: &PolicySpec| -> Vec<String> {
            document
                .bindings
                .iter()
                .map(|binding| binding.matcher.model.clone())
                .collect()
        };
        assert_eq!(models(&whole), ["a", "c", "e", "*"]);
        let own = settings.policy_document().expect("the settings' document");
        for picked in 0..(1_u32 << built.len()) {
            let subset: Vec<_> = built
                .iter()
                .enumerate()
                .filter(|(at, _)| picked & (1 << at) != 0)
                .map(|(_, label)| *label)
                .collect();
            let document = with_labels(&settings, &subset).expect("a subset builds");
            let of_subset = |binding: &&BindingSpec| {
                subset
                    .iter()
                    .any(|(model, _)| binding.matcher.model == *model)
            };
            let mut expected: Vec<_> = whole.bindings.iter().filter(of_subset).cloned().collect();
            expected.extend(own.bindings.iter().cloned());
            assert_eq!(document.bindings, expected, "{subset:?}");
            for (name, profile) in &document.profiles {
                assert_eq!(whole.profiles.get(name), Some(profile), "{subset:?}");
            }
        }
    }

    /// A label whose binding would pass a bound of the document is refused
    /// by its model and its policy, in the words of that bound. A key block
    /// names eight sources at most: the builder's own sentence, as for a
    /// main policy. The view lists 256 headers at most, which the
    /// controller's check holds for the whole document and for no model by
    /// name: a label that would list the 257th is refused here, and the
    /// settings' own document and every other label stand.
    #[test]
    fn a_label_past_a_bound_of_the_document_is_refused_by_name() {
        let headed = |count: usize| {
            let mut settings = config(at_flag_defaults("round_robin"));
            settings.routing_key_override.headers =
                (0..count).map(|at| format!("x-k{at}")).collect();
            settings
        };
        let eight = headed(8);
        let as_main = RoutingSettings {
            policy: at_flag_defaults("consistent_hashing"),
            ..eight.clone()
        }
        .policy_document()
        .expect_err("a ninth source of the ring's key");
        assert_eq!(
            with_labels(&eight, &[("m", "consistent_hashing")]),
            Err(label_error("m", "consistent_hashing", &as_main.to_string())),
        );
        assert!(with_labels(&headed(7), &[("m", "consistent_hashing")]).is_ok());

        // The manual rule names its headers by the rule's own fields, which
        // no key block bounds: its headers are the view's.
        let full = headed(255);
        let document =
            with_labels(&full, &[("b", "manual"), ("m", "manual")]).expect("256 headers");
        assert_eq!(document.view.headers.len(), 256);
        assert!(validated(&document).is_ok());
        let over = headed(256);
        let own = over.policy_document().expect("the settings' own document");
        assert!(own.view.headers.is_empty());
        let too_many = SpecError::TooManyHeaders {
            count: 257,
            max: 256,
        };
        assert_eq!(
            with_labels(&over, &[("a", "random"), ("m", "manual")]),
            Err(label_error("m", "manual", &too_many.to_string())),
        );
        assert_eq!(
            with_labels(&over, &[("a", "random")]).map(|document| document.bindings.len()),
            Ok(2)
        );
        // A list the settings' own policy took past the bound is the
        // settings': a label that adds no header is not refused for it.
        let own_over = RoutingSettings {
            policy: at_flag_defaults("manual"),
            ..headed(256)
        };
        let headers = |document: PolicySpec| document.view.headers.len();
        assert_eq!(own_over.policy_document().map(headers), Ok(257));
        assert_eq!(
            with_labels(&own_over, &[("m", "random")]).map(headers),
            Ok(257)
        );
    }

    /// Models in the order of their ids, and one profile for each policy:
    /// two models that name one policy share its profile, and the same
    /// labels give the same document.
    #[test]
    fn labelled_models_stand_in_id_order_and_share_a_policys_profile() {
        let settings = config(at_flag_defaults("round_robin"));
        let named = [
            ("zeta", "random"),
            ("alpha", "least_load"),
            ("mid", "random"),
        ];
        let document = with_labels(&settings, &named).expect("a labelled document");
        let bound: Vec<_> = document
            .bindings
            .iter()
            .map(|binding| (binding.matcher.model.as_str(), binding.profile.as_deref()))
            .collect();
        assert_eq!(
            bound,
            [
                ("alpha", Some("least_load")),
                ("mid", Some("random")),
                ("zeta", Some("random")),
                ("*", Some("round_robin")),
            ]
        );
        let profiles: Vec<_> = document.profiles.keys().map(String::as_str).collect();
        assert_eq!(profiles, ["least_load", "random", "round_robin"]);
        let again = [named[2], named[0], named[1]];
        assert_eq!(with_labels(&settings, &again), Ok(document.clone()));
        assert!(validated(&document).is_ok());
        assert_eq!(single_profile(&document, "mid"), "random");
        assert_eq!(single_profile(&document, "alpha"), "least_load");
        // A policy's profile is written once for each model that names it,
        // and a second write leaves it as the first left it.
        for name in PRESETS {
            let settings = config(main_beside(name));
            let once = with_labels(&settings, &[("a", name)]).expect("one model");
            let twice = with_labels(&settings, &[("a", name), ("b", name)]).expect("two models");
            assert_eq!(twice.bindings.len(), 3, "{name}");
            assert_eq!(twice.profiles, once.profiles, "{name}");
            assert_eq!(twice.view, once.view, "{name}");
            // Each labelled model reads the same index, and no other model
            // reads another than it read.
            for model in ["a", "b"] {
                assert_eq!(index_of(&twice, model), index_of(&once, "a"), "{name}");
            }
            let index = index_of(&twice, "another");
            assert_eq!(index, index_of(&once, "another"), "{name}");
            // The second model's binding is the first's but for its model:
            // its key sources and its session rule are written again.
            assert_eq!(twice.bindings[0], once.bindings[0], "{name}");
            let mut second = twice.bindings[1].clone();
            assert_eq!(second.matcher.model, "b", "{name}");
            second.matcher.model = "a".to_owned();
            assert_eq!(second, twice.bindings[0], "{name}");
        }
    }

    /// Two models that name one policy by their labels are bound to one
    /// profile, so a stage that keeps a history of its own keeps one for
    /// both. Round robin draws the start of a list it sees first from a
    /// count of its own. In one labelled document the second model's first
    /// turn starts where that count stands behind the first model's. In a
    /// document that labels it alone, it starts where the first model's
    /// did: a gateway's own registry builds a policy for each labelled
    /// model, so there every labelled model starts as the first.
    #[test]
    fn labelled_models_of_one_policy_share_what_its_stages_count() {
        use crate::engine::{
            framework::{Engine, Scratch},
            view::{Candidate, Constraints, Health, PoolFacts, RequestFacts, WorkerId},
        };
        let settings = config(at_flag_defaults("random"));
        let engine = |named: &[(&str, &str)]| {
            with_labels(&settings, named)
                .expect("a labelled document")
                .validate_with(&["x-smg-tenant-id"])
                .expect("the controller's check takes the document")
        };
        let pool: Vec<Candidate> = (0..32)
            .map(|position| Candidate {
                position,
                id: WorkerId(100 + u64::from(position)),
                key: 100 + position,
                health: Health {
                    healthy: true,
                    ..Health::default()
                },
                ..Candidate::default()
            })
            .collect();
        let first_turn = |engine: &Engine, model: &'static str| {
            let facts = RequestFacts {
                model,
                ..RequestFacts::default()
            };
            let decision = engine.decide(
                &facts,
                &PoolFacts::default(),
                &pool,
                &Constraints::default(),
                &mut Scratch::new(),
            );
            let pick = decision.pick().expect("a pick");
            pick.targets.iter().next().expect("a target").position
        };
        let both = engine(&[("a", "round_robin"), ("b", "round_robin")]);
        let a_first = first_turn(&both, "a");
        let b_behind_a = first_turn(&both, "b");
        let alone = engine(&[("b", "round_robin")]);
        let b_alone = first_turn(&alone, "b");
        assert_eq!(b_alone, a_first);
        assert_ne!(b_behind_a, b_alone);
    }

    /// No name holds two profiles. A pair's sides are named for the side,
    /// so a `bucket` label beside a pair whose prefill policy is `bucket`
    /// adds a profile of its own name, at the numbers a label builds it
    /// with, and leaves the side's at the settings'.
    #[test]
    fn a_labels_profile_stands_beside_a_side_of_the_same_policy() {
        let settings = RoutingSettings {
            mode: pair_mode(Some(at_flag_defaults("bucket")), None),
            ..config(at_flag_defaults("round_robin"))
        };
        let own = settings
            .policy_document()
            .expect("the pair mode's document");
        let document = with_labels(&settings, &[("m", "bucket")]).expect("a labelled one");
        let profiles: Vec<_> = document.profiles.keys().map(String::as_str).collect();
        assert_eq!(profiles, ["bucket", "decode", "prefill", "round_robin"]);
        for side in ["decode", "prefill", "round_robin"] {
            assert_eq!(document.profiles[side], own.profiles[side], "{side}");
        }
        let by_label = config(PolicySettings::of_label("bucket").expect("the label's policy"))
            .policy_document()
            .expect("the label's policy as a main policy");
        assert_eq!(document.profiles["bucket"], by_label.profiles["bucket"]);
        assert_ne!(
            document.profiles["bucket"].score,
            own.profiles["prefill"].score
        );
    }

    /// What a label changes outside its own profile and binding. Under
    /// settings that set nothing, `manual` alone changes the view, by its
    /// header. `cache_aware` alone writes the index block: its model's own
    /// entry, at the size and sweep a label builds it with. No label writes
    /// a signal. Under
    /// the sticky override beside a main policy of the hash ring, whose
    /// binding lists no header, a label the override places adds the
    /// override's headers to the view.
    #[test]
    fn a_label_changes_the_view_for_its_headers_alone() {
        let mut changed = Vec::new();
        for name in PRESETS {
            let settings = config(main_beside(name));
            let own = settings.policy_document().expect("the settings' document");
            let document = with_labels(&settings, &[("m", name)]).expect("a labelled one");
            assert_eq!(document.signals, own.signals, "{name}");
            assert_eq!(document.plans, own.plans, "{name}");
            if document.view != own.view {
                let mut view = own.view.clone();
                view.headers.clone_from(&document.view.headers);
                assert_eq!(document.view, view, "{name}: the headers alone");
                changed.push(("view", name));
            }
            if document.index != own.index {
                changed.push(("index", name));
            }
        }
        assert_eq!(changed, [("view", "manual"), ("index", "cache_aware")]);
        let labelled = with_labels(
            &config(at_flag_defaults("round_robin")),
            &[("m", "cache_aware")],
        )
        .expect("a labelled one");
        let index = IndexSpec {
            max_size: 10_000,
            sweep_s: 30,
            ..IndexSpec::default()
        };
        assert_eq!(
            labelled.index.into_iter().collect::<Vec<_>>(),
            [("m".to_owned(), index)]
        );

        let mut ring = config(at_flag_defaults("consistent_hashing"));
        ring.routing_key_override.enabled = true;
        let own = ring.policy_document().expect("the ring's document");
        assert!(own.view.headers.is_empty());
        for name in PLACED.into_iter().chain(["prefix_hash"]) {
            let document = with_labels(&ring, &[("m", name)]).expect("a labelled one");
            assert_eq!(document.view.headers, ["x-smg-routing-key"], "{name}");
        }
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
        let paired = pair_alone(&pairs).expect("the pair's own document");
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
        let pairs_alone = pair_alone(&clash);
        assert!(pairs_alone.is_ok(), "{pairs_alone:?}");
        assert_eq!(clash.policy_document(), pairs_alone);
        assert_eq!(many_routers(clash).policy_document(), pairs_alone);
    }

    /// No pair mode loses its document to its single request. Every policy
    /// is the main one in turn, the sides are unnamed or both of one policy,
    /// and the sticky override is off and on. Where the pair's own document
    /// is refused, the mode's is refused by the same sentence. Where the
    /// controller's check takes the pair's own document, it takes the
    /// mode's, which binds a single request behind the pair in every one.
    #[test]
    fn no_pair_mode_loses_its_document_to_its_single_request() {
        for main in PRESETS {
            for sides in std::iter::once(None).chain(PRESETS.iter().copied().map(Some)) {
                for sticky in [false, true] {
                    let mut settings = RoutingSettings {
                        mode: pair_mode(sides.map(at_flag_defaults), sides.map(at_flag_defaults)),
                        ..config(at_flag_defaults(main))
                    };
                    settings.routing_key_override.enabled = sticky;
                    let named = format!("{main} over {sides:?}, the override {sticky}");
                    let (alone, said) = (pair_alone(&settings), settings.policy_document());
                    let (Ok(own), Ok(both)) = (&alone, &said) else {
                        assert_eq!(said, alone, "{named}");
                        continue;
                    };
                    let taken = validated(own).is_ok();
                    assert_eq!(validated(both).is_ok(), taken, "{named}");
                    assert!(!taken || both.bindings.len() == 2, "{named}");
                }
            }
        }
    }
}
